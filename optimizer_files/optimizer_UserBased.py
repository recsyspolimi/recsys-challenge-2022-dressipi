import sys
import scipy.sparse as sps
import sys

import scipy.sparse as sps

sys.path.append('../')

from RecSys_Course_AT_PoliMi.Pipeline.recommender_tuning import hypertune
from RecSys_Course_AT_PoliMi.Pipeline.data_extraction import get_dataframes
from RecSys_Course_AT_PoliMi.Pipeline.matrices_creation import get_URM_split_val
from RecSys_Course_AT_PoliMi.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommenderStacked
from RecSys_Course_AT_PoliMi.Pipeline.optuna_utils import Integer

item_features_df, train_sessions_df, train_purchases_df, \
test_sessions_df, candidate_items_df = get_dataframes(path_to_dataset='../')

URM_train, URM_val_views, URM_val_purch, mapped_items_to_ignore, mapped_val_sessions_arr, val_session_mapping, item_mapping = get_URM_split_val(
    item_features_df=item_features_df,
    train_purchases_df=train_purchases_df,
    train_sessions_df=train_sessions_df,
    val_start_ts = '2021-05-01',
    val_end_ts = '2021-06-01',
    unique_interactions=True,
    abs_half_life=182,
    cyclic_decay=True,
    purch_weight=1,
    view_weight=0.2,
    score_graph=False,
)

URM_use_train = sps.vstack([URM_train, URM_val_views])
URM_fake_zeros = sps.csr_matrix((URM_train.shape[0], URM_train.shape[1]))
URM_use_validation = sps.vstack([URM_fake_zeros, URM_val_purch])

hyp =  {
    "topK": Integer(9000, 10000),
    "shrink": Integer(0, 2000),
    'normalize' : True,
    'feature_weighting' : 'BM25',
    "URM_bias": False,
    "use_implementation": "cython"
}

additional_recommender_args = {"URM_validation": URM_val_views}

best_params = hypertune(
    URM_train=URM_train,
    URM_val_views=URM_use_train,
    URM_val_purch=URM_use_validation,
    item_mapping=item_mapping,
    mapped_items_to_ignore=mapped_items_to_ignore,
    num_trials=200,
    recommender_class=UserKNNCFRecommenderStacked,
    additional_recommender_args=additional_recommender_args,
    hyperparameters_dict=hyp
)