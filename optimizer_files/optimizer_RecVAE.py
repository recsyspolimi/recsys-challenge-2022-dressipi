import sys

sys.path.append('../')

from RecSys_Course_AT_PoliMi.Pipeline.recommender_tuning import hypertune
from RecSys_Course_AT_PoliMi.Pipeline.data_extraction import get_dataframes
from RecSys_Course_AT_PoliMi.Pipeline.matrices_creation import get_URM_split_val
from RecSys_Course_AT_PoliMi.Recommenders.Neural.RecVAERecommender import RecVAERecommender
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
    "epochs": Integer(2, 50),
    "hidden_dim": Categorical([256, 512, 1024]),
    "latent_dim": Categorical([64, 128, 256, 512]),
    "gamma": Real(low=1e-5, high=0.01, prior='uniform'),
    "learning_rate": Real(low=1e-5, high=0.1, prior='uniform'),
    "batch_size": Categorical([1024, 2048, 4096, 8192, 16384]),
    "dropout": Real(low=0.01, high=0.8, prior='uniform'),
    "n_enc_epochs": Integer(2, 5),
    "n_dec_epochs": Integer(1, 3),
    "validation_every_n": 1,
    "validation_metric": "MRR",
    "lower_validations_allowed": 3
}


best_params = hypertune(
    URM_train=URM_train,
    URM_val_views=URM_val_views,
    URM_val_purch=URM_val_purch,
    item_mapping=item_mapping,
    mapped_items_to_ignore=mapped_items_to_ignore,
    num_trials=200,
    recommender_class=RecVAERecommender,
    hyperparameters_dict=hyp
)