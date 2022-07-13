import os

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

from DressipiChallenge.Pipeline.data_splitting import split_dataframes_val, split_dataframes_test
from DressipiChallenge.Pipeline.matrices_creation import create_ICM, create_csr_matrix
from DressipiChallenge.Pipeline.submission import create_submission
from DressipiChallenge.Pipeline.utils import batch_compute_item_score, compute_MRR, generate_predictions, flat_list, \
    count_saved_files, concatenate_predictions

tqdm.pandas()


def init_model(current_model, is_content_based, train_df, val_df, URM_train, URM_val, ICM):
    if is_content_based:
        # content based
        model = current_model(URM_train, ICM)
    elif current_model.RECOMMENDER_NAME == 'GRU4RecRecommender':
        # GRU4Rec
        model = current_model(
            train_df.copy(),
            val_df.copy(),
            URM_train,
        )
    elif current_model.RECOMMENDER_NAME == 'UserKNNCFRecommenderStackedXGBoost':
        # UserKNNCF
        model = current_model(
            URM_train,
            URM_val,
        )
    else:
        # collaborative
        model = current_model(URM_train)

    return model


def fit_model(model, model_hyp, mapped_items_to_ignore, URM_new, saved_model_path):
    model.set_items_to_ignore(mapped_items_to_ignore)

    if not os.path.exists(saved_model_path):
        os.makedirs(saved_model_path)
        model.fit(**model_hyp)
        model.save_model(saved_model_path, model.RECOMMENDER_NAME)
    else:
        if (not os.path.exists(os.path.join(saved_model_path, model.RECOMMENDER_NAME + '.zip'))) & \
                (not os.path.exists(os.path.join(saved_model_path, model.RECOMMENDER_NAME + '.pkl'))) & \
                (not os.path.exists(os.path.join(saved_model_path, model.RECOMMENDER_NAME + '.pth'))):
            model.fit(**model_hyp)
            model.save_model(saved_model_path, model.RECOMMENDER_NAME)
        else:
            model.load_model(saved_model_path, model.RECOMMENDER_NAME)

    model.set_items_to_ignore(mapped_items_to_ignore)
    model.set_URM_train(URM_new)

    return model


def integrate_predictions(reranked_df, fitted_rec_for_integration, test_sessions_arr, cutoff = 100):
    covered_sessions = set(reranked_df.session_id.to_list())
    sessions_to_integrate = set([session for session in test_sessions_arr if session not in covered_sessions])

    to_integrate_df = generate_predictions(
        models=[fitted_rec_for_integration],
        session_ids=list(sessions_to_integrate),
        add_item_score=False,
        cutoff=cutoff
    )[0]

    return concatenate_predictions([reranked_df, to_integrate_df])


def rename_columns(df, models):
    i = 0

    for column in df.columns:
        if not (column in ['session_id', 'item_id', 'target', 'is_fake']):
            df.rename(
                columns={column: models[i].RECOMMENDER_NAME + '_score'}, inplace=True)
            i += 1

    return df


def _get_gb_candidates(session_ids, models, val_purchases=None, cutoff=100, is_training=False):
    # for each model, generate a dataframe with a row per candidate per session
    dataframes_list = generate_predictions(models, session_ids, cutoff=cutoff, add_item_score=True)
    # merge the dataframes keeping the scores of each model on a separate column
    # and dropping duplicate session-item combinations
    boosted_df = dataframes_list[0]
    current_name = models[0].RECOMMENDER_NAME
    boosted_df.rename(columns={'item_score': current_name + '_score'}, inplace=True)
    for i in range(1, len(dataframes_list)):
        current_name = models[i].RECOMMENDER_NAME
        dataframes_list[i].rename(columns={'item_score': current_name + '_score'}, inplace=True)
        boosted_df = pd.merge(boosted_df, dataframes_list[i], how='outer', on=['session_id', 'item_id'])

    merged_df = boosted_df.drop_duplicates(keep='first')

    if is_training:
        # add missing purchases from the validation set and add target column
        val_purchases = val_purchases[['session_id', 'item_id']]
        val_purchases.loc[:, 'target'] = 1
        merged_df = pd.merge(merged_df, val_purchases, how='outer', on=['session_id', 'item_id'], indicator='is_fake')
        is_fake_dict = {'left_only': False, 'right_only': True, 'both': False}
        merged_df['is_fake'] = merged_df['is_fake'].map(is_fake_dict).astype(bool)
        merged_df.rename_axis('index', inplace=True)
        merged_df.sort_values(by=['session_id', 'index'], inplace=True, na_position='first')
        merged_df.reset_index(inplace=True, drop=True)
        merged_df['target'] = merged_df['target'].fillna(0, inplace=False).astype('uint8')
    else:
        merged_df.rename_axis('index', inplace=True)
        merged_df.sort_values(by=['session_id', 'index'], inplace=True, na_position='first')
        merged_df.reset_index(inplace=True, drop=True)

    # fill missing item_scores for each model (therefore for each score column)
    i = 0

    filled_df = merged_df.copy()

    print('Filling missing item_scores...')

    for column in filled_df.columns:
        if not (column in ['session_id', 'item_id', 'target', 'is_fake']):
            # select rows where the current score column is NaN
            selected_df = filled_df[filled_df[column].isna()].copy()

            session_ids = np.unique(selected_df.session_id.to_list())
            items_to_compute = np.unique(selected_df.item_id.to_list())

            if (len(session_ids) == 0) | (len(items_to_compute) == 0):
                continue

            # compute scores for ALL session_id and item_id combinations in selected_df (there will be some useless ones)
            scores_list = batch_compute_item_score(models[i], session_ids, items_to_compute, n_batch=100)
            scores_list = np.array(scores_list)

            item_lengths = selected_df.groupby(['session_id']).size().to_list()
            item_indices = selected_df.item_id.to_list()
            session_indices = flat_list([[id] * length for id, length in zip(range(len(session_ids)), item_lengths)])

            # index over the scores and only take the ones requested in selected_df
            score_col = scores_list[session_indices, item_indices]

            # insert missing scores into filled_df
            selected_df[column] = score_col
            filled_df.fillna(selected_df, inplace=True)

            i += 1

    # filled_df = rename_columns(filled_df, models)

    if is_training:
        return filled_df.drop(columns='is_fake'), filled_df[filled_df.is_fake == False].reset_index(drop=True)

    return filled_df


def remove_useless_cols(candidates_df, true_candidates_df, features_dict):
    del_keys = []
    for key, value in features_dict.items():
        if value == 0:
            del_keys.append(key)
    for key in del_keys:
        features_dict.pop(key)

    cols_to_keep = list(features_dict.keys())
    cols_to_keep.extend(['session_id', 'item_id'])

    features_to_drop = [col for col in candidates_df.columns.to_list() if col not in cols_to_keep]
    # TODO remove highly correlated features
    candidates_df = candidates_df.drop(columns=features_to_drop)

    true_candidates_df = true_candidates_df.drop(columns=features_to_drop)

    return candidates_df, true_candidates_df


def get_useless_cols(df, max_correlation=0.95, alpha=0.05):
    """
    Retrieve columns that are highly correlated
    """
    useless_cols = []
    columns = df.columns
    for i, column in enumerate(columns):
        vals = df[column].unique()
        if len(vals) < 2:
            useless_cols.append(column)
        else:
            for column2 in columns[i + 1:]:
                r, p = pearsonr(df[column].to_numpy(), df[column2].to_numpy())
                if abs(r) >= max_correlation and p <= alpha:
                    useless_cols.append(column)
                    break
    return useless_cols


def get_features_to_drop(cols_to_keep, session_attributes_df, item_attributes_df):
    cols_to_keep.extend(['session_id', 'item_id'])
    features = []
    features.extend(session_attributes_df.columns.to_list())
    features.extend(item_attributes_df.columns.to_list())
    features_to_drop = [col for col in features if col not in cols_to_keep]

    return features_to_drop


def print_MRR_score(reranked_df, val_purch_df):
    val_purch_df['target'] = True
    reranked_df = pd.merge(reranked_df, val_purch_df[['session_id', 'item_id', 'target']], how='left',
                           on=['session_id', 'item_id'])
    reranked_df['target'] = reranked_df['target'].fillna(False)

    MRR_score = compute_MRR(reranked_df)
    print('MRR score: ' + str(MRR_score))


def GB_insert_item_feature(pred_df, item_feature_df):
    if 'item_id' in item_feature_df.columns:
        return pd.merge(pred_df, item_feature_df, how='left', on='item_id')
    else:  # ICM format
        return pd.merge(pred_df, item_feature_df, how='left', left_on='item_id', right_index=True)


def GB_insert_session_feature(pred_df, session_feature_df):
    if 'session_id' in session_feature_df.columns:
        return pd.merge(pred_df, session_feature_df, how='left', on='session_id')
    else:  # UCM format
        return pd.merge(pred_df, session_feature_df, how='left', left_on='session_id', right_index=True)


def compute_nonzero_candidates(df, column_name):
    scores = np.array(df[column_name].to_list())
    return len(scores[scores != 0])


def filter_cold_sessions(candidates_df, min_num_nonzero_candidates):
    print("Filtering cold sessions...")
    sessions_to_remove = set()
    for col in tqdm(candidates_df.columns.to_list()):
        if col not in ['session_id', 'item_id', 'target', 'is_fake']:
            nonzero_df = candidates_df.groupby('session_id').apply(
                lambda x: compute_nonzero_candidates(x, col)
            )
            sessions_to_remove.update(nonzero_df[nonzero_df < min_num_nonzero_candidates].index.to_list())
    return candidates_df[~candidates_df.session_id.isin(sessions_to_remove)].reset_index(drop=True)


def load_gb_train_df(
        candidates_train_df_path='./Dataset/xgb_candidates/candidates_train_df.parquet',
        true_candidates_train_df_path='./Dataset/xgb_candidates/true_candidates_train_df.parquet',
        session_ids=None, val_purchases=None, models=None, cutoff=100, min_num_nonzero_candidates=None):
    """
    Returns a dataframe with a varying number of candidate items per session and a dataframe containing the target (1
    if the candidate is the item purchased in the session)
    Arguments:
    * session_ids: list of session ids to be considered
    * val_purchases: dataframe containing all validation purchases
    * models: list of FITTED recommender objects to be used to produce candidates
    * cutoff: number of candidates to generate per model

    """

    if not os.path.exists(candidates_train_df_path) or not os.path.exists(true_candidates_train_df_path):
        new_path = candidates_train_df_path
        path_components = new_path.split('/')
        new_path = new_path.replace(path_components[-1], '')

        if not os.path.exists(new_path):
            # Create a new directory because it does not exist
            os.makedirs(new_path)

        print(
            '[LOAD] candidates_train_df not found in path, creating new one in "' + candidates_train_df_path + '"')

        candidates_df, true_candidates_df = _get_gb_candidates(session_ids=session_ids, val_purchases=val_purchases,
                                                               models=models, cutoff=cutoff, is_training=True)

        candidates_df.to_parquet(candidates_train_df_path)
        true_candidates_df.to_parquet(true_candidates_train_df_path)

    else:
        candidates_df = pd.read_parquet(candidates_train_df_path)
        true_candidates_df = pd.read_parquet(true_candidates_train_df_path)
        print('[LOAD] loaded train candidates')

    if min_num_nonzero_candidates is not None:
        candidates_df = filter_cold_sessions(candidates_df, min_num_nonzero_candidates)
        true_candidates_df = filter_cold_sessions(true_candidates_df, min_num_nonzero_candidates)

    return candidates_df, true_candidates_df


def load_gb_test_df(candidates_test_df_path='./Dataset/xgb_candidates/candidates_test_df.parquet',
                    session_ids=None, models=None, cutoff=100, min_num_nonzero_candidates=None):
    """
    Returns a dataframe with a varying number of candidate items per session and a dataframe containing the target (1
    if the candidate is the item purchased in the session)
    Arguments:
    * session_ids: list of session ids to be considered
    * models: list of FITTED recommender objects to be used to produce candidates
    * cutoff: number of candidates to generate per model

    """

    if not os.path.exists(candidates_test_df_path):
        new_path = candidates_test_df_path
        path_components = new_path.split('/')
        new_path = new_path.replace(path_components[-1], '')

        if not os.path.exists(new_path):
            # Create a new directory because it does not exist
            os.makedirs(new_path)

        print(
            '[LOAD] candidates_test_df not found in path, creating new one in "' + candidates_test_df_path + '"')

        candidates_df = _get_gb_candidates(session_ids=session_ids, models=models, cutoff=cutoff)

        candidates_df.to_parquet(candidates_test_df_path)

    else:
        candidates_df = pd.read_parquet(candidates_test_df_path)
        print('[LOAD] loaded test candidates')

    if min_num_nonzero_candidates is not None:
        candidates_df = filter_cold_sessions(candidates_df, min_num_nonzero_candidates)

    return candidates_df


def GB_split_val(original_candidates_df_path, train_size=0.8):
    candidates_df = pd.read_parquet(original_candidates_df_path)

    gss = GroupShuffleSplit(n_splits=1, train_size=train_size)
    train_idx, test_idx = next(gss.split(candidates_df, groups=candidates_df['session_id']))

    train_candidates_df = candidates_df.iloc[train_idx]
    test_candidates_df = candidates_df.iloc[test_idx]

    return train_candidates_df, test_candidates_df


def GB_process_train_dataset(
        train_sessions_df, train_purchases_df, item_features_df,
        session_attributes_train_df, item_attributes_df,
        model_dict,
        val_start_ts, val_end_ts,
        candidates_train_df_path, true_candidates_train_df_path,
        min_num_nonzero_candidates, num_candidates_per_model=100,
        saved_model_train_path='./save/saved_model_train/'
):
    assert (isinstance(model_dict['URM_params'], list))
    assert ((len(model_dict['URM_params']) == 1) | (len(model_dict['URM_params']) == len(model_dict['model_classes'])))
    assert (len(model_dict['models_hyp']) == len(model_dict['model_classes']))
    assert (len(model_dict['models_hyp']) == len(model_dict['is_content_based']))

    # create mapping and split the dataset in training and validation
    if (len(model_dict['URM_params']) == 1): #  | (count_saved_files(saved_model_train_path) == len(model_dict['model_classes']))

        weighted_train_set_df, val_purch_df, val_views_df, train_session_mapping, val_session_mapping, item_mapping, \
        mapped_items_to_ignore_val, val_sessions_arr = split_dataframes_val(
            train_sessions_df.copy(), train_purchases_df.copy(), item_features_df.copy(),
            val_start_ts=val_start_ts, val_end_ts=val_end_ts,
            **model_dict['URM_params'][0]
        )
    else:
        recommender_args = []
        for URM_params in model_dict['URM_params']:
            weighted_train_set_df, val_purch_df, val_views_df, train_session_mapping, val_session_mapping, item_mapping, \
            mapped_items_to_ignore_val, val_sessions_arr = split_dataframes_val(
                train_sessions_df.copy(), train_purchases_df.copy(), item_features_df.copy(),
                val_start_ts=val_start_ts, val_end_ts=val_end_ts,
                **URM_params
            )
            recommender_args.append({
                'weighted_train_set_df': weighted_train_set_df,
                'val_purch_df': val_purch_df,
                'val_views_df': val_views_df
            })

    if not os.path.exists(candidates_train_df_path) or not os.path.exists(true_candidates_train_df_path):
        models = []

        # create ICM
        ICM = None
        if True in model_dict['is_content_based']:
            ICM, _ = create_ICM(item_features_df.copy(), item_mapping)

        if len(model_dict['URM_params']) == 1:
            # create URM
            URM_train = create_csr_matrix(weighted_train_set_df.copy(), len(train_session_mapping), len(item_mapping))
            URM_val = create_csr_matrix(val_views_df.copy(), len(val_session_mapping), len(item_mapping))

            for i in range(len(model_dict['model_classes'])):
                current_model = init_model(model_dict['model_classes'][i], model_dict['is_content_based'][i],
                                           weighted_train_set_df.copy(), val_views_df.copy(),
                                           URM_train, URM_val, ICM)

                fitted_model = fit_model(current_model, model_dict['models_hyp'][i], mapped_items_to_ignore_val,
                                         URM_val, saved_model_train_path)
                models.append(fitted_model)

        else:
            for i in range(len(model_dict['model_classes'])):
                # print("Fitting " + model_dict['model_classes'][i].RECOMMENDER_NAME + "...")
                URM_train = create_csr_matrix(recommender_args[i]['weighted_train_set_df'].copy(),
                                              len(train_session_mapping), len(item_mapping))
                URM_val = create_csr_matrix(recommender_args[i]['val_views_df'].copy(), len(val_session_mapping),
                                            len(item_mapping))

                current_model = init_model(model_dict['model_classes'][i], model_dict['is_content_based'][i],
                                           recommender_args[i]['weighted_train_set_df'].copy(),
                                           recommender_args[i]['val_views_df'].copy(), URM_train, URM_val, ICM)

                fitted_model = fit_model(current_model, model_dict['models_hyp'][i], mapped_items_to_ignore_val,
                                         URM_val, saved_model_train_path)
                models.append(fitted_model)

        candidates_df, true_candidates_df = load_gb_train_df(session_ids=val_sessions_arr,
                                                             val_purchases=val_purch_df.copy(), models=models,
                                                             cutoff=num_candidates_per_model,
                                                             candidates_train_df_path=candidates_train_df_path,
                                                             true_candidates_train_df_path=true_candidates_train_df_path,
                                                             min_num_nonzero_candidates=min_num_nonzero_candidates)
    else:
        candidates_df, true_candidates_df = load_gb_train_df(session_ids=None, val_purchases=None, models=None,
                                                             cutoff=num_candidates_per_model,
                                                             candidates_train_df_path=candidates_train_df_path,
                                                             true_candidates_train_df_path=true_candidates_train_df_path,
                                                             min_num_nonzero_candidates=min_num_nonzero_candidates)

    # add features
    session_attributes_train_df['session_id'] = session_attributes_train_df['session_id'].map(
        val_session_mapping)  # candidates_df uses VAL mapping, as the command set_URM(URM_val) has been executed
    item_attributes_df['item_id'] = item_attributes_df['item_id'].map(item_mapping)
    print('Attributes mapped')

    candidates_df = GB_insert_session_feature(candidates_df.copy(), session_attributes_train_df.copy())
    candidates_df = GB_insert_item_feature(candidates_df.copy(), item_attributes_df.copy())

    true_candidates_df = GB_insert_session_feature(true_candidates_df.copy(), session_attributes_train_df.copy())
    true_candidates_df = GB_insert_item_feature(true_candidates_df.copy(), item_attributes_df.copy())

    print('The total number of attributes is:' + str(len(list(candidates_df.columns))))
    print('Attributes in use are:')
    print(candidates_df.columns)

    target_df = candidates_df[['target']]
    candidates_df = candidates_df.drop(columns='target')

    '''
    if (num_candidates_per_model < 100) | (min_num_nonzero_candidates is not None):
        assert (all(candidates_df.groupby('session_id').size().values >= 100))
        assert (all(true_candidates_df.groupby('session_id').size().values >= 100))
    '''

    return candidates_df, target_df, true_candidates_df, val_purch_df


def GB_process_inference_dataset(
        train_sessions_df, train_purchases_df, item_features_df, candidate_items_df, test_sessions_df,
        session_attributes_test_df, item_attributes_df,
        model_dict,
        features_to_drop,
        candidates_df_path,
        min_num_nonzero_candidates,
        num_candidates_per_model=100,
        saved_model_inference_path='./save/saved_model_inference/'
):
    assert (isinstance(model_dict['URM_params'], list))
    assert ((len(model_dict['URM_params']) == 1) | (len(model_dict['URM_params']) == len(model_dict['model_classes'])))
    assert (len(model_dict['models_hyp']) == len(model_dict['model_classes']))
    assert (len(model_dict['models_hyp']) == len(model_dict['is_content_based']))

    # crete mapping
    if (len(model_dict['URM_params']) == 1): # | (count_saved_files(saved_model_inference_path) == len(model_dict['model_classes'])):

        weighted_train_set_df, test_sessions_df, train_session_mapping, test_session_mapping, item_mapping, mapped_items_to_ignore, test_sessions_arr = \
            split_dataframes_test(
                train_sessions_df.copy(), train_purchases_df.copy(), item_features_df.copy(), candidate_items_df.copy(),
                test_sessions_df.copy(), **model_dict['URM_params'][0])
    else:
        recommender_args = []
        for URM_params in model_dict['URM_params']:
            weighted_train_set_df, test_set_df, train_session_mapping, test_session_mapping, item_mapping, mapped_items_to_ignore, test_sessions_arr = \
                split_dataframes_test(
                    train_sessions_df.copy(), train_purchases_df.copy(), item_features_df.copy(),
                    candidate_items_df.copy(), test_sessions_df.copy(), **URM_params)
            recommender_args.append({
                'weighted_train_set_df': weighted_train_set_df,
                'test_sessions_df': test_set_df,
            })

    if not os.path.exists(candidates_df_path):
        models = []

        # create ICM
        ICM = None
        if True in model_dict['is_content_based']:
            ICM, _ = create_ICM(item_features_df.copy(), item_mapping)

        if len(model_dict['URM_params']) == 1:
            # create URM
            URM_all = create_csr_matrix(weighted_train_set_df.copy(), len(train_session_mapping), len(item_mapping))
            URM_test = create_csr_matrix(test_sessions_df.copy(), len(test_session_mapping), len(item_mapping))

            for i in range(len(model_dict['model_classes'])):
                current_model = init_model(model_dict['model_classes'][i], model_dict['is_content_based'][i],
                                           weighted_train_set_df.copy(), test_sessions_df.copy(),
                                           URM_all, URM_test, ICM)

                fitted_model = fit_model(current_model, model_dict['models_hyp'][i], mapped_items_to_ignore,
                                         URM_test, saved_model_inference_path)
                models.append(fitted_model)

        else:
            for i in range(len(model_dict['model_classes'])):
                # print("Fitting " + model_dict['model_classes'][i].RECOMMENDER_NAME + "...")
                URM_all = create_csr_matrix(recommender_args[i]['weighted_train_set_df'].copy(),
                                            len(train_session_mapping), len(item_mapping))
                URM_test = create_csr_matrix(recommender_args[i]['test_sessions_df'].copy(), len(test_session_mapping),
                                             len(item_mapping))

                current_model = init_model(model_dict['model_classes'][i], model_dict['is_content_based'][i],
                                           recommender_args[i]['weighted_train_set_df'].copy(),
                                           recommender_args[i]['test_sessions_df'].copy(), URM_all, URM_test, ICM)

                fitted_model = fit_model(current_model, model_dict['models_hyp'][i], mapped_items_to_ignore,
                                         URM_test, saved_model_inference_path)
                models.append(fitted_model)

        # generate candidates
        candidates_df = load_gb_test_df(session_ids=test_sessions_arr, models=models, cutoff=num_candidates_per_model,
                                        candidates_test_df_path=candidates_df_path,
                                        min_num_nonzero_candidates=min_num_nonzero_candidates)

    else:
        candidates_df = load_gb_test_df(session_ids=None, models=None, cutoff=num_candidates_per_model,
                                        candidates_test_df_path=candidates_df_path,
                                        min_num_nonzero_candidates=min_num_nonzero_candidates)

    # add features
    session_attributes_test_df['session_id'] = session_attributes_test_df['session_id'].map(test_session_mapping)
    item_attributes_df['item_id'] = item_attributes_df['item_id'].map(item_mapping)
    print('Attributes mapped')

    candidates_df = GB_insert_session_feature(candidates_df.copy(), session_attributes_test_df.copy())
    candidates_df = GB_insert_item_feature(candidates_df.copy(), item_attributes_df.copy())

    print('The total number of attributes is:' + str(len(list(candidates_df.columns))))
    print('Attributes in use are:')
    print(candidates_df.columns)

    # drop useless features
    candidates_df = candidates_df.drop(columns=features_to_drop)

    '''
    if (num_candidates_per_model < 100) | (min_num_nonzero_candidates is not None):
        assert (all(candidates_df.groupby('session_id').size().values >= 100))
    '''

    return candidates_df, test_session_mapping, item_mapping


def GB_rerank_candidates(candidates_df, predictions, cutoff=None):
    scores = []
    for a in predictions.values:
        scores.extend(a)

    candidates_df['score'] = scores
    candidates_df['session_id'] = candidates_df['session_id'].astype('Int64')

    reranked_df = candidates_df.sort_values(by=['session_id', 'score'], inplace=False, ascending=[True, False])

    if cutoff is not None:
        reranked_df = reranked_df.groupby('session_id').head(cutoff)

    return reranked_df


def GB_integrate_pathological(
        train_sessions_df, train_purchases_df, item_features_df, test_sessions_df, candidate_items_df,
        reranked_df,
        recommender, model_hyp, cutoff=100):

    weighted_train_set_df, test_sessions_df, train_session_mapping, test_session_mapping, item_mapping, mapped_items_to_ignore, test_sessions_arr = \
        split_dataframes_test(
            train_sessions_df.copy(), train_purchases_df.copy(), item_features_df.copy(), candidate_items_df.copy(),
            test_sessions_df.copy())

    ICM, _ = create_ICM(item_features_df.copy(), item_mapping=item_mapping)
    URM_all = create_csr_matrix(weighted_train_set_df.copy(), len(train_session_mapping), len(item_mapping))
    URM_test = create_csr_matrix(test_sessions_df.copy(), len(test_session_mapping), len(item_mapping))

    rec_cold = recommender(URM_all, ICM)
    rec_cold.fit(**model_hyp)

    rec_cold.set_items_to_ignore(mapped_items_to_ignore)
    rec_cold.set_URM_train(URM_test)

    integrated_df = integrate_predictions(reranked_df=reranked_df, fitted_rec_for_integration=rec_cold,
                                          test_sessions_arr=test_sessions_arr, cutoff=cutoff)

    prediction_df = create_submission(prediction_df=integrated_df.copy(),
                                      item_mapping=item_mapping, session_mapping=test_session_mapping, cutoff=100)

    return prediction_df