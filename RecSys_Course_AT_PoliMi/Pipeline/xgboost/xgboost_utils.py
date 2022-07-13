import os

import xgboost as xgb
import numpy as np
import pandas as pd

from RecSys_Course_AT_PoliMi.Pipeline.submission import create_submission
from RecSys_Course_AT_PoliMi.Pipeline.xgboost.xgboost_tuning import XGB_hypertune
from RecSys_Course_AT_PoliMi.Pipeline.gradient_boosting_utils import GB_process_train_dataset, \
    remove_useless_cols, print_MRR_score, GB_process_inference_dataset, GB_rerank_candidates, get_features_to_drop, \
    GB_integrate_pathological


def XGB_train(candidates_df, target_df, xgb_hyperparams, num_boost_round, save_path='./Dataset/xgb_model/'):
    dcandidates = xgb.DMatrix(
        candidates_df.drop(columns=['session_id', 'item_id']),
        label=target_df,
        qid=candidates_df['session_id'],
        nthread=-1,
        missing=np.NaN,
        # group = candidates_df.groupby(['session_id']).size().to_list(),
    )
    xgb_model = xgb.train(
        xgb_hyperparams,
        dcandidates,
        num_boost_round=num_boost_round,
        callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)],
    )

    if not os.path.exists(save_path):
        # Create a new directory because it does not exist
        os.makedirs(save_path)

    xgb_model.save_model(os.path.join(save_path, 'xgb_model.json'))

    return xgb_model


def XGB_rerank(candidates_df, xgb_model, cutoff=100):
    print('Reranking...')
    predictions = candidates_df.groupby('session_id').progress_apply(
        lambda x: xgb_model.predict(
            xgb.DMatrix(
                x.drop(columns=['session_id', 'item_id']),
                nthread=-1,
                missing=np.NaN,
            )
        )
    )
    reranked_df = GB_rerank_candidates(candidates_df.copy(), predictions.copy(), cutoff=cutoff)

    return reranked_df


def XGB_tune_test(
        train_sessions_df, train_purchases_df, item_features_df,
        session_attributes_train_df, item_attributes_df,
        model_dict,
        retrain_after_dropping_features=False,
        num_candidates_per_model=100, cutoff=100,
        val_start_ts='2021-05-01', val_end_ts='2021-06-01',
        candidates_train_df_path="./Dataset/xgb_candidates/candidates_train_df.parquet",
        true_candidates_train_df_path="./Dataset/xgb_candidates/true_candidates_train_df.parquet",
        xgb_model_save_path="./Dataset/xgb_model/",
        reranked_df_save_path='./Dataset/xgb_candidates/reranked_df.parquet',
        min_num_nonzero_candidates=None,
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
        min_num_nonzero_candidates=min_num_nonzero_candidates,
    )

    if not os.path.exists(xgb_model_save_path):
        # tune the boosting algorithm
        xgb_hyperparams, iteration = XGB_hypertune(candidates_df=candidates_df.copy(), target_df=target_df.copy(), **hypertune_params)

        # train the boosting algorithm with optimal parameters
        xgb_model = XGB_train(candidates_df=candidates_df.copy(), target_df=target_df.copy(), xgb_hyperparams=xgb_hyperparams,
                              num_boost_round=iteration, save_path=xgb_model_save_path)

        # compute feature importance
        feature_importance = xgb_model.get_score()
        print("Feature importance: " + str(feature_importance))

        if retrain_after_dropping_features:
            # Remove useless columns and retrain (useless if feature importance=0)
            candidates_df, true_candidates_df = remove_useless_cols(candidates_df.copy(), true_candidates_df.copy(),
                                                                    feature_importance)

            xgb_model = XGB_train(candidates_df=candidates_df.copy(), target_df=target_df.copy(), xgb_hyperparams=xgb_hyperparams,
                                  num_boost_round=iteration, save_path=xgb_model_save_path)
    else:
        # load pre trained model
        xgb_model = xgb.Booster()
        xgb_model.load_model(os.path.join(xgb_model_save_path, 'xgb_model.json'))
        if retrain_after_dropping_features:
            _, true_candidates_df = remove_useless_cols(candidates_df.copy(), true_candidates_df.copy(), xgb_model.get_score())

    # compute prediction and print MRR score at the end of the training process
    true_candidates_df = true_candidates_df.drop(columns=['target', 'is_fake'])
    reranked_df = XGB_rerank(candidates_df=true_candidates_df.copy(), xgb_model=xgb_model, cutoff=cutoff)
    print_MRR_score(reranked_df.copy(), val_purch_df.copy())

    if reranked_df_save_path is not None:
        reranked_df.to_parquet(reranked_df_save_path)

    return xgb_model, reranked_df


def XGB_train_test(
        train_sessions_df, train_purchases_df, item_features_df,
        session_attributes_train_df, item_attributes_df,
        xgb_hyperparams, iteration, model_dict,
        retrain_after_dropping_features=False,
        num_candidates_per_model=100, cutoff=100,
        val_start_ts='2021-05-01', val_end_ts='2021-06-01',
        candidates_train_df_path="./Dataset/xgb_candidates/candidates_train_df.parquet",
        true_candidates_train_df_path="./Dataset/xgb_candidates/true_candidates_train_df.parquet",
        xgb_model_save_path="./Dataset/xgb_model/",
        reranked_df_save_path='./Dataset/xgb_candidates/reranked_df.parquet',
        min_num_nonzero_candidates=None,
):
    # pd.options.mode.chained_assignment = None

    # build the dataframes with the candidates needed for the boosting algorithm
    candidates_df, target_df, true_candidates_df, val_purch_df = GB_process_train_dataset(
        train_sessions_df=train_sessions_df.copy(), train_purchases_df=train_purchases_df.copy(), item_features_df=item_features_df.copy(),
        session_attributes_train_df=session_attributes_train_df.copy(), item_attributes_df=item_attributes_df.copy(),
        model_dict=model_dict, num_candidates_per_model=num_candidates_per_model,
        val_start_ts=val_start_ts, val_end_ts=val_end_ts,
        candidates_train_df_path=candidates_train_df_path, true_candidates_train_df_path=true_candidates_train_df_path,
        min_num_nonzero_candidates=min_num_nonzero_candidates,
    )

    if not os.path.exists(xgb_model_save_path):
  
        # train the boosting algorithm with optimal parameters
        xgb_model = XGB_train(candidates_df=candidates_df.copy(), target_df=target_df.copy(), xgb_hyperparams=xgb_hyperparams,
                              num_boost_round=iteration, save_path=xgb_model_save_path)

        # compute feature importance
        feature_importance = xgb_model.get_score()
        print("Feature importance: " + str(feature_importance))

        if retrain_after_dropping_features:
            # Remove useless columns and retrain (useless if feature importance=0)
            candidates_df, true_candidates_df = remove_useless_cols(candidates_df.copy(), true_candidates_df.copy(), feature_importance)
            
            xgb_model = XGB_train(candidates_df=candidates_df.copy(), target_df=target_df.copy(), xgb_hyperparams=xgb_hyperparams,
                                  num_boost_round=iteration, save_path=xgb_model_save_path)
    else:
        # load pre trained model
        xgb_model = xgb.Booster()
        xgb_model.load_model(os.path.join(xgb_model_save_path, 'xgb_model.json'))
        if retrain_after_dropping_features:
            _, true_candidates_df = remove_useless_cols(candidates_df.copy(), true_candidates_df.copy(), xgb_model.get_score())

    # compute prediction and print MRR score at the end of the training process
    op_true_candidates_df = true_candidates_df.drop(columns=['target', 'is_fake'])
    reranked_df = XGB_rerank(candidates_df=op_true_candidates_df.copy(), xgb_model=xgb_model, cutoff=cutoff)
    print_MRR_score(reranked_df.copy(), val_purch_df.copy())

    if reranked_df_save_path is not None:
        reranked_df.to_parquet(reranked_df_save_path)

    return xgb_model, true_candidates_df, reranked_df


def XGB_test_submission(
        train_sessions_df, train_purchases_df, item_features_df, test_sessions_df, candidate_items_df,
        session_attributes_test_df, item_attributes_df,
        xgb_model,
        model_dict, num_candidates_per_model=100, cutoff=100,
        retrain_after_dropping_features=False,
        candidates_df_path="./Dataset/xgb_candidates/candidates_test_df.parquet",
        min_num_nonzero_candidates=None,
):
    # compute the features to drop
    features_to_drop = []
    if retrain_after_dropping_features:
        cols_to_keep = xgb_model.feature_names
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
    reranked_df = XGB_rerank(candidates_df=candidates_df.copy(), xgb_model=xgb_model, cutoff=cutoff)
    prediction_df = create_submission(prediction_df=reranked_df.copy(), item_mapping=item_mapping, session_mapping=test_session_mapping, cutoff=cutoff)

    return prediction_df


def XGB_test_submission_with_integration(
        train_sessions_df, train_purchases_df, item_features_df, test_sessions_df, candidate_items_df,
        session_attributes_test_df, item_attributes_df,
        xgb_model,
        model_dict, cold_recommender, cold_model_hyp,
        num_candidates_per_model=100, cutoff=100,
        retrain_after_dropping_features=False,
        candidates_df_path="./Dataset/xgb_candidates/candidates_test_df.parquet",
        min_num_nonzero_candidates=None,
):
    # compute the features to drop
    features_to_drop = []
    if retrain_after_dropping_features:
        cols_to_keep = xgb_model.feature_names
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
    reranked_df = XGB_rerank(candidates_df=candidates_df.copy(), xgb_model=xgb_model, cutoff=cutoff)

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