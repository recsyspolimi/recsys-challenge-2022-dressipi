import sys
import pandas as pd

sys.path.append('..')
sys.path.append('/Dataset')

from Pipeline.recommender_tuning import hypertune
from Pipeline.data_extraction import get_dataframes
from Pipeline.matrices_creation import get_URM_split_val, create_URM
from DressipiChallenge.Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask
from Pipeline.optuna_utils import Integer, Real, Categorical

item_features_df, train_sessions_df, train_purchases_df, test_sessions_df, candidate_items_df = get_dataframes()

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
    "epochs": Categorical([500]),
    "learning_rate": Real(low=1e-5, high=1e-2, prior="log-uniform"),
    "l2_reg": Real(low=1e-6, high=1e-2, prior="log-uniform"),
    "dropout": Real(low=0., high=0.8, prior="uniform"),
    "total_anneal_steps": Integer(100000, 600000),
    "anneal_cap": Real(low=0., high=0.6, prior="uniform"),
    "batch_size": Categorical([128, 256, 512, 1024]),

    "encoding_size": Integer(1, min(512, 23690)),
    "next_layer_size_multiplier": Integer(2, 10),
    "max_n_hidden_layers": Integer(1, 4),

    # Constrain the model to a maximum number of parameters so that its size does not exceed 7 GB
    # Estimate size by considering each parameter uses float32
    "max_parameters": Categorical([7*1e9*8/32]),
    "validation_every_n": 4,
    "validation_metric": "MRR",
    "lower_validations_allowed": 4
}

best_params = hypertune(
    URM_train=URM_train,
    URM_val_views=URM_val_views,
    URM_val_purch=URM_val_purch,
    item_mapping=item_mapping,
    mapped_items_to_ignore=mapped_items_to_ignore,
    num_trials=300,
    recommender_class=MultVAERecommender_OptimizerMask,
    hyperparameters_dict=hyp
)
