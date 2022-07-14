import os

import lightgbm as lgb
import pandas as pd

from RecSys_Course_AT_PoliMi.Pipeline.submission import create_submission
from RecSys_Course_AT_PoliMi.Pipeline.lightgbm.lightgbm_tuning import LGB_hypertune
from RecSys_Course_AT_PoliMi.Pipeline.gradient_boosting_utils import GB_process_inference_dataset, print_MRR_score, \
    remove_useless_cols, GB_process_train_dataset, GB_rerank_candidates, get_features_to_drop, GB_integrate_pathological


def LGB_train(candidates_df, target_df, lgb_hyperparams, num_boost_round, save_path='./Dataset/lgb_model/'):
    input_df = candidates_df.drop(columns=['session_id', 'item_id'])
    dcandidates = lgb.Dataset(
        input_df,
        label=target_df,
        group=candidates_df.groupby('session_id').size().to_list(),
        feature_name=input_df.columns.to_list(),
        categorical_feature=[i for i in range(73, 141)]
    )
    lgb_model = lgb.train(
        lgb_hyperparams,
        dcandidates,
        num_boost_round=num_boost_round,
        callbacks=[lgb.callback.log_evaluation(show_stdv=True)],
    )

    if not os.path.exists(save_path):
        # Create a new directory because it does not exist
        os.makedirs(save_path)

    lgb_model.save_model(os.path.join(save_path, 'lgb_model.json'))

    return lgb_model


def LGB_rerank(candidates_df, lgb_model, cutoff=100):
    print('Reranking...')
    predictions = candidates_df.groupby('session_id').progress_apply(
        lambda x: lgb_model.predict(
            x.drop(columns=['session_id', 'item_id']),
        )
    )
    reranked_df = GB_rerank_candidates(candidates_df.copy(), predictions.copy(), cutoff=cutoff)

    return reranked_df


def LGB_tune_test(
        train_sessions_df, train_purchases_df, item_features_df,
        session_attributes_train_df, item_attributes_df,
        model_dict,
        retrain_after_dropping_features=False,
        num_candidates_per_model=100, cutoff=100,
        val_start_ts='2021-05-01', val_end_ts='2021-06-01',
        candidates_train_df_path="./Dataset/xgb_candidates/candidates_train_df.parquet",
        true_candidates_train_df_path="./Dataset/xgb_candidates/true_candidates_train_df.parquet",
        lgb_model_save_path="./Dataset/lgb_model/",
        reranked_df_save_path='./Dataset/xgb_candidates/lgb_reranked_df.parquet',
        min_num_nonzero_candidates=None,
        keep_unpredicted=False,
        **hypertune_params
):
    # pd.options.mode.chained_assignment = None

    # build the dataframes with the candidates needed for the boosting algorithm
    candidates_df, target_df, true_candidates_df, val_purch_df = GB_process_train_dataset(
        train_sessions_df=train_sessions_df.copy(), train_purchases_df=train_purchases_df.copy(), item_features_df=item_features_df.copy(),
        session_attributes_train_df=session_attributes_train_df.copy(), item_attributes_df=item_attributes_df.copy(),
        model_dict=model_dict, num_candidates_per_model=num_candidates_per_model,
        val_start_ts=val_start_ts, val_end_ts=val_end_ts,
        candidates_train_df_path=candidates_train_df_path, true_candidates_train_df_path=true_candidates_train_df_path,
        min_num_nonzero_candidates=min_num_nonzero_candidates, keep_unpredicted=keep_unpredicted
    )

    if not os.path.exists(lgb_model_save_path):
        # tune the boosting algorithm
        lgb_hyperparams, iteration = LGB_hypertune(candidates_df=candidates_df.copy(), target_df=target_df.copy(), **hypertune_params)

        # train the boosting algorithm with optimal parameters
        lgb_model = LGB_train(candidates_df=candidates_df.copy(), target_df=target_df.copy(), lgb_hyperparams=lgb_hyperparams,
                              num_boost_round=iteration, save_path=lgb_model_save_path)

        # compute feature importance
        zip_features = zip(lgb_model.feature_name(), lgb_model.feature_importance())
        feature_importance = dict(zip_features)
        print("Feature importance: " + str(feature_importance))

        if retrain_after_dropping_features:
            # Remove useless columns and retrain (useless if feature importance=0)
            candidates_df, true_candidates_df = remove_useless_cols(candidates_df.copy(), true_candidates_df.copy(), feature_importance)

            lgb_model = LGB_train(candidates_df=candidates_df.copy(), target_df=target_df.copy(), lgb_hyperparams=lgb_hyperparams,
                                  num_boost_round=iteration, save_path=lgb_model_save_path)
    else:
        # load pre trained model
        lgb_model = lgb.Booster(model_file=os.path.join(lgb_model_save_path, 'lgb_model.json'))
        if retrain_after_dropping_features:
            zip_features = zip(lgb_model.feature_name(), lgb_model.feature_importance())
            feature_importance = dict(zip_features)
            _, true_candidates_df = remove_useless_cols(candidates_df.copy(), true_candidates_df.copy(), feature_importance)

    # compute prediction and print MRR score at the end of the training process
    true_candidates_df = true_candidates_df.drop(columns=['target', 'is_fake'])
    reranked_df = LGB_rerank(candidates_df=true_candidates_df.copy(), lgb_model=lgb_model, cutoff=cutoff)
    print_MRR_score(reranked_df.copy(), val_purch_df.copy())

    if reranked_df_save_path is not None:
        reranked_df.to_parquet(reranked_df_save_path)

    return lgb_model, reranked_df


def LGB_train_test(
        train_sessions_df, train_purchases_df, item_features_df,
        session_attributes_train_df, item_attributes_df,
        lgb_hyperparams, iteration, model_dict,
        retrain_after_dropping_features=False,
        num_candidates_per_model=100, cutoff=100,
        val_start_ts='2021-05-01', val_end_ts='2021-06-01',
        candidates_train_df_path="./Dataset/xgb_candidates/candidates_train_df.parquet",
        true_candidates_train_df_path="./Dataset/xgb_candidates/true_candidates_train_df.parquet",
        lgb_model_save_path="./Dataset/lgb_model/",
        reranked_df_save_path='./Dataset/xgb_candidates/lgb_reranked_df.parquet',
        min_num_nonzero_candidates=None,
        keep_unpredicted=False
):
    # pd.options.mode.chained_assignment = None

    # build the dataframes with the candidates needed for the boosting algorithm
    candidates_df, target_df, true_candidates_df, val_purch_df = GB_process_train_dataset(
        train_sessions_df=train_sessions_df.copy(), train_purchases_df=train_purchases_df.copy(), item_features_df=item_features_df.copy(),
        session_attributes_train_df=session_attributes_train_df.copy(), item_attributes_df=item_attributes_df.copy(),
        model_dict=model_dict, num_candidates_per_model=num_candidates_per_model,
        val_start_ts=val_start_ts, val_end_ts=val_end_ts,
        candidates_train_df_path=candidates_train_df_path, true_candidates_train_df_path=true_candidates_train_df_path,
        min_num_nonzero_candidates=min_num_nonzero_candidates, keep_unpredicted=keep_unpredicted
    )

    if not os.path.exists(lgb_model_save_path):
        # train the boosting algorithm with optimal parameters
        lgb_model = LGB_train(candidates_df=candidates_df.copy(), target_df=target_df.copy(), lgb_hyperparams=lgb_hyperparams,
                              num_boost_round=iteration, save_path=lgb_model_save_path)

        # compute feature importance
        zip_features = zip(lgb_model.feature_name(), lgb_model.feature_importance())
        feature_importance = dict(zip_features)
        print("Feature importance: " + str(feature_importance))

        if retrain_after_dropping_features:
            # Remove useless columns and retrain (useless if feature importance=0)
            candidates_df, true_candidates_df = remove_useless_cols(candidates_df.copy(), true_candidates_df.copy(), feature_importance)

            lgb_model = LGB_train(candidates_df=candidates_df.copy(), target_df=target_df.copy(), lgb_hyperparams=lgb_hyperparams,
                                  num_boost_round=iteration, save_path=lgb_model_save_path)

    else:
        # load pre trained model
        lgb_model = lgb.Booster(model_file=os.path.join(lgb_model_save_path, 'lgb_model.json'))
        if retrain_after_dropping_features:
            zip_features = zip(lgb_model.feature_name(), lgb_model.feature_importance())
            feature_importance = dict(zip_features)
            _, true_candidates_df = remove_useless_cols(candidates_df.copy(), true_candidates_df.copy(), feature_importance)

    # compute prediction and print MRR score at the end of the training process
    op_true_candidates_df = true_candidates_df.drop(columns=['target', 'is_fake'])
    reranked_df = LGB_rerank(candidates_df=op_true_candidates_df.copy(), lgb_model=lgb_model, cutoff=cutoff)
    print_MRR_score(reranked_df.copy(), val_purch_df.copy())

    if reranked_df_save_path is not None:
        reranked_df.to_parquet(reranked_df_save_path)

    return lgb_model, true_candidates_df, reranked_df


def LGB_test_submission(
        train_sessions_df, train_purchases_df, item_features_df, test_sessions_df, candidate_items_df,
        session_attributes_test_df, item_attributes_df,
        lgb_model,
        model_dict, num_candidates_per_model=100, cutoff=100,
        retrain_after_dropping_features=False,
        candidates_df_path="./Dataset/xgb_candidates/candidates_test_df.parquet",
        min_num_nonzero_candidates=None,
):
    # compute the features to drop
    features_to_drop = []
    if retrain_after_dropping_features:
        cols_to_keep = lgb_model.feature_name()
        features_to_drop = get_features_to_drop(cols_to_keep, session_attributes_test_df.copy(), item_attributes_df.copy())

    # build the dataframe with the candidates needed for the boosting algorithm
    candidates_df, test_session_mapping, item_mapping = GB_process_inference_dataset(
        train_sessions_df=train_sessions_df.copy(), train_purchases_df=train_purchases_df.copy(), item_features_df=item_features_df.copy(),
        candidate_items_df=candidate_items_df.copy(), test_sessions_df=test_sessions_df.copy(),
        session_attributes_test_df=session_attributes_test_df.copy(), item_attributes_df=item_attributes_df.copy(),
        model_dict=model_dict, num_candidates_per_model=num_candidates_per_model,
        features_to_drop=features_to_drop,
        candidates_df_path=candidates_df_path,
        min_num_nonzero_candidates=min_num_nonzero_candidates,
    )

    assert(all(candidates_df.groupby('session_id').size().values >= cutoff))

    # generate prediction and create submission csv
    reranked_df = LGB_rerank(candidates_df=candidates_df.copy(), lgb_model=lgb_model, cutoff=cutoff)
    prediction_df = create_submission(prediction_df=reranked_df.copy(), item_mapping=item_mapping, session_mapping=test_session_mapping, cutoff=cutoff)

    return prediction_df


def LGB_test_submission_with_integration(
        train_sessions_df, train_purchases_df, item_features_df, test_sessions_df, candidate_items_df,
        session_attributes_test_df, item_attributes_df,
        lgb_model,
        model_dict, cold_recommender, cold_model_hyp,
        num_candidates_per_model=100, cutoff=100,
        retrain_after_dropping_features=False,
        candidates_df_path="./Dataset/xgb_candidates/candidates_test_df.parquet",
        min_num_nonzero_candidates=None,
):
    # compute the features to drop
    features_to_drop = []
    if retrain_after_dropping_features:
        cols_to_keep = lgb_model.feature_name()
        features_to_drop = get_features_to_drop(cols_to_keep, session_attributes_test_df.copy(), item_attributes_df.copy())

    # build the dataframe with the candidates needed for the boosting algorithm
    candidates_df, test_session_mapping, item_mapping = GB_process_inference_dataset(
        train_sessions_df=train_sessions_df.copy(), train_purchases_df=train_purchases_df.copy(), item_features_df=item_features_df.copy(),
        candidate_items_df=candidate_items_df.copy(), test_sessions_df=test_sessions_df.copy(),
        session_attributes_test_df=session_attributes_test_df.copy(), item_attributes_df=item_attributes_df.copy(),
        model_dict=model_dict, num_candidates_per_model=num_candidates_per_model,
        features_to_drop=features_to_drop,
        candidates_df_path=candidates_df_path,
        min_num_nonzero_candidates=min_num_nonzero_candidates,
    )

    assert(all(candidates_df.groupby('session_id').size().values >= cutoff))

    # generate prediction
    reranked_df = LGB_rerank(candidates_df=candidates_df.copy(), lgb_model=lgb_model, cutoff=cutoff)

    # integrate pathological sessions and create submission csv
    prediction_df = GB_integrate_pathological(
        train_sessions_df=train_sessions_df.copy(),
        train_purchases_df=train_purchases_df.copy(),
        item_features_df=item_features_df.copy(),
        test_sessions_df=test_sessions_df.copy(),
        candidate_items_df=candidate_items_df.copy(),
        reranked_df=reranked_df.copy(),
        recommender=cold_recommender,
        model_hyp=cold_model_hyp,
        cutoff=cutoff
    )

    return prediction_df

