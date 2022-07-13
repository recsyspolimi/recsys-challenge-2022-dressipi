import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm.auto import tqdm
import time

from RecSys_Course_AT_PoliMi.Evaluation.Evaluator import EvaluatorHoldout
from RecSys_Course_AT_PoliMi.Evaluation.metrics import MRR
from RecSys_Course_AT_PoliMi.Utils.seconds_to_biggest_unit import seconds_to_biggest_unit

tqdm.pandas()


def merge_views_purchases(train_sessions_df, train_purchases_df):
    train_set_df = pd.concat([train_sessions_df, train_purchases_df])
    train_set_df.sort_values(by=['session_id', 'date'], ascending=[True, True], inplace=True)
    train_set_df.reset_index(drop=True, inplace=True)

    return train_set_df


def get_items_to_exclude(item_features_df, candidate_item_ids):
    """
    WARNING: candidate_item_ids MUST NOT be mapped!
    """
    train_item_ids = item_features_df['item_id'].unique()
    non_candidate_mask = np.isin(train_item_ids, candidate_item_ids, invert=True)
    return train_item_ids[non_candidate_mask]  # return non-candidate items


def get_mapped_sessions_to_recommend(test_sessions_df, session_mapping):
    test_session_ids = test_sessions_df['session_id'].unique()
    mapped_test_session_ids = [session_mapping[elem] for elem in test_session_ids]
    return np.sort(mapped_test_session_ids)


def create_mapping(df_col, return_inverse_mapping=False):
    """
    Create new mapping for indexes such that the maximum ID corresponds to the number of unique ID values
    """
    sorted_col = np.sort(df_col.unique())
    mapping = {k: v for k, v in zip(sorted_col, range(len(sorted_col)))}

    if return_inverse_mapping:
        return mapping, sorted_col  # e.g. sorted_col[new_index]=original_id (np.array)

    return mapping


def create_candidate_set(test_df):
    return test_df['item_id'].unique()


def batch_recommend(recommender, user_id_arr, n_batch=1, return_scores=False, **recommender_args):
    batch_size = len(user_id_arr) // n_batch

    user_id_batch = [[] for _ in range(n_batch)]

    for i in range(n_batch):
        if i == n_batch - 1:
            user_id_batch[i] = user_id_arr[batch_size * i:]
        else:
            user_id_batch[i] = user_id_arr[batch_size * i:batch_size * (i + 1)]

    ranking_lists = []
    score_lists = []

    print('Recommending...')

    for iter in tqdm(range(n_batch)):

        if return_scores:
            rankings, scores = recommender.recommend(user_id_batch[iter], return_scores=True,
                                                     **recommender_args)
            # Take #cutoff biggest elements from each sublist in 'scores' and sort them in descending order
            scores = [
                sorted(np.partition(a, len(a) - recommender_args['cutoff'])[-recommender_args['cutoff']:], reverse=True)
                for a in scores]
            score_lists.extend(scores)

        else:
            rankings = recommender.recommend(user_id_batch[iter], return_scores=False, **recommender_args)

        ranking_lists.extend(rankings)

    assert (all([len(a) == recommender_args['cutoff'] for a in ranking_lists]))
    print('Done!')

    if return_scores:
        return ranking_lists, score_lists
    else:
        return ranking_lists


def batch_compute_item_score(recommender, user_id_arr, items_to_compute, n_batch=1):
    batch_size = len(user_id_arr) // n_batch

    user_id_batch = [[] for _ in range(n_batch)]

    for i in range(n_batch):
        if i == n_batch - 1:
            user_id_batch[i] = user_id_arr[batch_size * i:]
        else:
            user_id_batch[i] = user_id_arr[batch_size * i:batch_size * (i + 1)]

    score_list = []

    print('Computing item scores...')

    for iter in tqdm(range(n_batch)):
        scores = recommender._compute_item_score(user_id_array=user_id_batch[iter], items_to_compute=items_to_compute)
        score_list.extend(scores)

    print('Done!')

    return score_list


def generate_predictions(models, session_ids, add_item_score=True, cutoff=100):
    dataframes_list = []

    for model in models:
        if add_item_score:
            ranking_lists, score_lists = batch_recommend(model, session_ids,
                                                         n_batch=100,
                                                         cutoff=cutoff,
                                                         remove_seen_flag=True,
                                                         remove_custom_items_flag=True,
                                                         return_scores=True
                                                         )

            replicated_session_ids = [([session] * cutoff)
                                      for session in session_ids]

            session_list_col = flat_list(replicated_session_ids)
            ranking_list_col = flat_list(ranking_lists)
            score_list_col = flat_list(score_lists)

            ranking_df = pd.DataFrame(
                {'session_id': session_list_col, 'item_id': ranking_list_col, 'item_score': score_list_col})

        else:
            ranking_lists = batch_recommend(model, session_ids,
                                            n_batch=10,
                                            cutoff=cutoff,
                                            remove_seen_flag=True,
                                            remove_custom_items_flag=True,
                                            return_scores=False
                                            )

            replicated_session_ids = [([session] * cutoff)
                                      for session in session_ids]
            session_list_col = flat_list(replicated_session_ids)
            ranking_list_col = flat_list(ranking_lists)

            ranking_df = pd.DataFrame(
                {'session_id': session_list_col, 'item_id': ranking_list_col})

        dataframes_list.append(ranking_df)

    return dataframes_list

def concatenate_predictions(pred_df_list):
    return pd.concat(pred_df_list).rename_axis('index').sort_values(['session_id', 'index']).reset_index(drop=True)

def generate_hybrid_predictions(rec, rec_cold, cold_sessions, non_cold_sessions, cutoff = 100):
    pred_df = generate_predictions(
        models=[rec],
        session_ids=non_cold_sessions,
        add_item_score=False,
        cutoff=cutoff
    )[0]
    pred_df_cold = generate_predictions(
        models=[rec_cold],
        session_ids=cold_sessions,
        add_item_score=False,
        cutoff=cutoff
    )[0]
    
    return concatenate_predictions([pred_df, pred_df_cold])

def count_saved_files(path):
    count = 0
    if os.path.exists(path):
        count = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])

    print("Saved files found: " + str(count))
    return count


def flat_list(list_of_lists):
    flat_list = []

    for i in range(len(list_of_lists)):
        flat_list.extend(list_of_lists[i])

    return flat_list


def compute_MRR(predictions_df):
    predictions_df['target'] = predictions_df['target'].astype(bool)
    score = MRR()

    print('Computing MRR score...')
    predictions_df.groupby('session_id').progress_apply(
        lambda x: score.add_recommendations(x.target.to_list())
    )

    return score.get_metric_value()


def single_train_test_model(recommender_class, URM_train, URM_val_views, URM_val_purch, mapped_items_to_ignore,
                            item_mapping, hyperparameters_dict, additional_recommender_args={}, ignore_user_to_optimize = None):
    to_use_dict = additional_recommender_args.copy()

    # if "ICM_train" in additional_recommender_args:
    #     to_use_dict['ICM_train'] = filter_ICM(additional_recommender_args['ICM_train'].copy(), item_mapping)
    #     #print(to_use_dict['ICM_train'])

    rec = recommender_class(URM_train, **to_use_dict)

    eval = EvaluatorHoldout(URM_val_purch, [100], ignore_items=mapped_items_to_ignore, ignore_users=ignore_user_to_optimize)
    if hasattr(rec, "_run_epoch") and "validation_every_n" in hyperparameters_dict:

        # Training in early stopping mode
        epochs_previous = hyperparameters_dict['epochs'] if 'epochs' in hyperparameters_dict else None
        if epochs_previous is not None:
            del hyperparameters_dict['epochs']
            rec.fit(epochs=1, **hyperparameters_dict)
            hyperparameters_dict['epochs'] = epochs_previous 
        epoch, result = train_optuna_with_earlystopping(URM_train, URM_val_views, rec, hyperparameters_dict, eval)
        hyperparameters_dict['epochs'] = epoch
        return result, None

    rec.fit(**hyperparameters_dict)
    rec.set_URM_train(URM_val_views)  # TODO: does not work correctly with user-user similarities
    return eval.evaluateRecommender(rec)


def train_optuna_with_earlystopping(URM_train, URM_val_views, recommender, hyperparameters_dict, evaluator_object = None):

    epochs_min = 0
    stop_on_validation = True
    algorithm_name = "[HYPERTUNING]"

    epochs_max = hyperparameters_dict["epochs"] if "epochs" in hyperparameters_dict else 1000
    validation_every_n = hyperparameters_dict["validation_every_n"] if "validation_every_n" in hyperparameters_dict else 1
    validation_metric = hyperparameters_dict["validation_metric"] if "validation_metric" in hyperparameters_dict else "MRR"
    lower_validations_allowed = hyperparameters_dict["lower_validations_allowed"] if "lower_validations_allowed" in hyperparameters_dict else 5 

    assert epochs_max >= 0, "{}: Number of epochs_max must be >= 0, passed was {}".format(algorithm_name, epochs_max)
    assert epochs_min >= 0, "{}: Number of epochs_min must be >= 0, passed was {}".format(algorithm_name, epochs_min)
    assert epochs_min <= epochs_max, "{}: epochs_min must be <= epochs_max, passed are epochs_min {}, epochs_max {}".format(algorithm_name, epochs_min, epochs_max)

    # Train for max number of epochs with no validation nor early stopping
    # OR Train for max number of epochs with validation but NOT early stopping
    # OR Train for max number of epochs with validation AND early stopping
    assert evaluator_object is None or\
            (evaluator_object is not None and not stop_on_validation and validation_every_n is not None and validation_metric is not None) or\
            (evaluator_object is not None and stop_on_validation and validation_every_n is not None and validation_metric is not None and lower_validations_allowed is not None),\
        "{}: Inconsistent parameters passed, please check the supported uses".format(algorithm_name)


    start_time = time.time()

    recommender.best_validation_metric = None
    lower_validatons_count = 0
    convergence = False

    recommender.epochs_best = 0

    epochs_current = 0

    best_results_run = None

    while epochs_current < epochs_max and not convergence:

        recommender._run_epoch(epochs_current)

        # If no validation required, always keep the latest
        if evaluator_object is None:

            recommender.epochs_best = epochs_current

        # Determine whether a validaton step is required
        elif (epochs_current + 1) % validation_every_n == 0:

            print("{}: Validation begins...".format(algorithm_name))

            recommender._prepare_model_for_validation()

            # Change URM 
            recommender.set_URM_train(URM_val_views) 

            # If the evaluator validation has multiple cutoffs, choose the first one
            results_run, results_run_string = evaluator_object.evaluateRecommender(recommender)
            current_metric_value = results_run.iloc[0][validation_metric]

            print("{}: {}".format(algorithm_name, results_run_string))

            # Update optimal model
            assert np.isfinite(current_metric_value), "{}: metric value is not a finite number, terminating!".format(recommender.RECOMMENDER_NAME)

            if recommender.best_validation_metric is None or recommender.best_validation_metric < current_metric_value:

                best_results_run = results_run
                print("{}: New best model found! Updating.".format(algorithm_name))
                recommender.best_validation_metric = current_metric_value
                recommender._update_best_model()

                recommender.epochs_best = epochs_current +1
                lower_validatons_count = 0

            else:
                lower_validatons_count += 1


            if stop_on_validation and lower_validatons_count >= lower_validations_allowed and epochs_current >= epochs_min:
                convergence = True

                elapsed_time = time.time() - start_time
                new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

                best_epoch = epochs_current+1

                print("{}: Convergence reached! Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} {}".format(
                    algorithm_name, epochs_current+1, validation_metric, recommender.epochs_best, recommender.best_validation_metric, new_time_value, new_time_unit))
                
            # Change URM back
            recommender.set_URM_train(URM_train) 

        elapsed_time = time.time() - start_time
        new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

        print("{}: Epoch {} of {}. Elapsed time {:.2f} {}".format(
            algorithm_name, epochs_current+1, epochs_max, new_time_value, new_time_unit))

        epochs_current += 1

        sys.stdout.flush()
        sys.stderr.flush()

    # If no validation required, keep the latest
    if evaluator_object is None:

        recommender._prepare_model_for_validation()
        recommender._update_best_model()


    # Stop when max epochs reached and not early-stopping
    if not convergence:
        elapsed_time = time.time() - start_time
        new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

        if evaluator_object is not None and recommender.best_validation_metric is not None:
            print("{}: Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} {}".format(
                algorithm_name, epochs_current, validation_metric, recommender.epochs_best, recommender.best_validation_metric, new_time_value, new_time_unit))
        else:
            print("{}: Terminating at epoch {}. Elapsed time {:.2f} {}".format(
                algorithm_name, epochs_current, new_time_value, new_time_unit))
    
    return recommender.epochs_best, best_results_run


def filter_ICM(ICM_df, item_associations, csr=True):
    ICM_new = ICM_df

    if len(item_associations) > 0:
        ICM_new = ICM_df.loc[ICM_df.index.isin(item_associations.keys())]
        ICM_new.index = ICM_new.index.map(item_associations)

    if csr:
        ICM_new = sp.csr_matrix(ICM_new)
    return ICM_new


def is_cold_df(train_df, item_list):
    df = train_df.groupby('item_id').count().reset_index()[['item_id', 'session_id']]
    df['session_id'] = True
    item_df = pd.DataFrame(item_list, columns=['item_id'])
    item_df = pd.merge(right=item_df, left=df, on='item_id', how="right").fillna(False).rename(columns={'session_id':'is_cold'})
    return item_df


def get_cold_sessions(train_set_df, val_views_df, item_mapping, item_features_df):
    all_items = set(item_features_df.item_id.map(item_mapping).to_list())
    hot_items = set(train_set_df.item_id.to_list())
    cold_items = set([item for item in all_items if item not in hot_items])
    is_cold_df = val_views_df.groupby('session_id').apply(
        lambda x: set(x.item_id.to_list()) <= cold_items
    )
    mapped_cold_sessions = is_cold_df[is_cold_df == True].index.to_list()
    mapped_non_cold_sessions = is_cold_df[is_cold_df == False].index.to_list()

    return mapped_cold_sessions, mapped_non_cold_sessions

def get_one_view_sessions(train_set_df, val_views_df, item_mapping, item_features_df):
    one_interaction_users = val_views_df.groupby('session_id').count()
    one_interaction_users = one_interaction_users.loc[one_interaction_users.item_id <= 1]
    one_interaction_users = one_interaction_users.index.values

    cold_session, _ = get_cold_sessions(train_set_df, val_views_df, item_mapping, item_features_df)
    one_interaction_users = [x for x in one_interaction_users if x not in cold_session]
    
    return one_interaction_users

def normalize_groupwise(df):

    def min_max_normalize(x):
        for col in x.columns.to_list():
            if ('_score' in col) & ('TopPop' not in col):
                x[col] = (x[col]-x[col].min())/(x[col].max()-x[col].min())
        return x
        
    print('Applying min-max normalization...')

    df = df.groupby('session_id').progress_apply(
        lambda x: min_max_normalize(x)
    )
    return df