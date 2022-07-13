import sys

sys.path.append('../')

from RecSys_Course_AT_PoliMi.Pipeline.recommender_tuning import hypertune
from RecSys_Course_AT_PoliMi.Pipeline.data_extraction import get_dataframes
from RecSys_Course_AT_PoliMi.Pipeline.matrices_creation import get_URM_split_val
from RecSys_Course_AT_PoliMi.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from RecSys_Course_AT_PoliMi.Pipeline.optuna_utils import Integer, Real, Categorical

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
    'topK': Integer(10, 1000),
    'alpha': Real(0.5, 1.1),
    'beta': Real(0.1, 0.6),
    'normalize_similarity': Categorical([True, False]),
}

best_params = hypertune(
    URM_train=URM_train,
    URM_val_views=URM_val_views,
    URM_val_purch=URM_val_purch,
    item_mapping=item_mapping,
    mapped_items_to_ignore=mapped_items_to_ignore,
    num_trials=200,
    recommender_class=RP3betaRecommender,
    hyperparameters_dict=hyp
)
