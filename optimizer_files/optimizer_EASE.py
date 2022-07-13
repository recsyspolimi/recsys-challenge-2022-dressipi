import sys

sys.path.append('../')

from RecSys_Course_AT_PoliMi.Pipeline.recommender_tuning import hypertune
from RecSys_Course_AT_PoliMi.Pipeline.data_extraction import get_dataframes
from RecSys_Course_AT_PoliMi.Pipeline.matrices_creation import get_URM_split_val
from RecSys_Course_AT_PoliMi.Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from RecSys_Course_AT_PoliMi.Pipeline.optuna_utils import Real, Categorical

item_features_df, train_sessions_df, train_purchases_df, \
test_sessions_df, candidate_items_df = get_dataframes(path_to_dataset='../')

URM_train, URM_val_views, URM_val_purch, mapped_items_to_ignore, mapped_val_sessions_arr, val_session_mapping, item_mapping = get_URM_split_val(
    item_features_df=item_features_df,
    train_purchases_df=train_purchases_df,
    train_sessions_df=train_sessions_df,
    val_start_ts='2021-05-01',
    val_end_ts='2021-06-01',
    unique_interactions=True,
    abs_half_life=None,
    cyclic_decay=False,
    purch_weight=1,
    view_weight=1,
    score_graph=False,
)

hyp = {
    "topK": Categorical([None]),
    "normalize_matrix": Categorical([False, True]),
    "l2_norm": Real(low=1e0, high=1e7, prior='log-uniform'),
}

best_params = hypertune(
    URM_train=URM_train,
    URM_val_views=URM_val_views,
    URM_val_purch=URM_val_purch,
    item_mapping=item_mapping,
    mapped_items_to_ignore=mapped_items_to_ignore,
    num_trials=200,
    recommender_class=EASE_R_Recommender,
    hyperparameters_dict=hyp
)
