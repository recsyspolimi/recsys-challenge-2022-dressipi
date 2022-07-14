import sys

import numpy as np

sys.path.append('../')

from RecSys_Course_AT_PoliMi.Pipeline.lightgbm.lightgbm_utils import LGB_tune_test, LGB_test_submission
from RecSys_Course_AT_PoliMi.Pipeline.data_extraction import get_dataframes, load_attributes

from RecSys_Course_AT_PoliMi.Recommenders.NonPersonalizedRecommender import TopPop
from RecSys_Course_AT_PoliMi.Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from RecSys_Course_AT_PoliMi.Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from RecSys_Course_AT_PoliMi.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommenderStackedXGBoost
from RecSys_Course_AT_PoliMi.Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask
from RecSys_Course_AT_PoliMi.Recommenders.Neural.RecVAERecommender import RecVAERecommender
from RecSys_Course_AT_PoliMi.Recommenders.SessionBased.GRU4RecRecommender import GRU4RecRecommender

'''
This script generates the candidates to be re-ranked, tunes the booster, and generates the submission csv
'''

# DATASET LOADING
item_features_df, train_sessions_df, train_purchases_df, test_sessions_df, candidate_items_df = get_dataframes(final=True)

# ATTRIBUTES LOADING
session_attributes_train_df, session_attributes_test_df, item_attributes_df = load_attributes()

# SETUP
model_classes = [EASE_R_Recommender, TopPop, GRU4RecRecommender, ItemKNN_CFCBF_Hybrid_Recommender,
                 UserKNNCFRecommenderStackedXGBoost, RecVAERecommender, MultVAERecommender_OptimizerMask]
is_content_based = [False, False, False, True, False, False, False]
models_hyp = [
    # EASE_R_Recommender
    {'topK': None, 'normalize_matrix': False, 'l2_norm': 46.90311309040724},
    # TopPop
    {},
    # GRU4Rec
    {'loss': 'bpr-max', 'final_act': 'leaky', 'hidden_act': 'linear', 'final_leaky_alpha': 0.1836895839246784, 'hidden_leaky_alpha': 0.007070008525727627, 'final_elu_alpha': 1.3997047273946337, 'hidden_elu_alpha': 0.7314840907690104, 'batch_size': 32, 'dropout_p_embed': 0.05040587501172002, 'learning_rate': 0.026213731644977353, 'n_units': 122, 'n_epochs': 6, 'momentum': 0.11045778520191754, 'embedding': True, 'n_embedding': 98, 'n_sample': 2048, 'sample_alpha': 0.4550148464032252, 'bpreg': 0.4947513248456839, 'logq': 0.3522263282997991, 'init_as_normal': False},
    # ItemKNN_CFBF
    {'topK': 3953, 'shrink': 3097, 'similarity': 'cosine', 'feature_weighting': 'TF-IDF', 'normalize': True, 'ICM_weight': 0.08821918678995819},
    # UserKNN
    {'topK': 9531, 'shrink': 724, 'normalize': True, 'feature_weighting': 'BM25', 'URM_bias': False},
    # RecVAE
    {'epochs': 16, 'hidden_dim': 512, 'latent_dim': 512, 'gamma': 0.002120200481439875, 'learning_rate': 0.0008527025871297427, 'batch_size': 8192, 'dropout': 0.44365543009435443, 'n_enc_epochs': 5, 'n_dec_epochs': 1},
    # MultVAE
    {'epochs': 53, 'learning_rate': 0.00020953471872049278, 'l2_reg': 8.782615138235317e-06, 'dropout': 0.5565555861318344, 'total_anneal_steps': 244578, 'anneal_cap': 0.25108629040696107, 'batch_size': 128, 'encoding_size': 147, 'next_layer_size_multiplier': 2, 'max_n_hidden_layers': 2, 'max_parameters': 1750000000.0}
]

URM_params = [
    {'view_weight': 0.5, 'abs_half_life': 182, 'cyclic_decay': True},  # EASE_R_Recommender
    {},  # TopPop
    {'unique_interactions': False},  # GRU4Rec
    {'view_weight': 0.5, 'abs_half_life': 182, 'cyclic_decay': True},  # ItemKNN_CFBF
    {'view_weight': 0.2, 'abs_half_life': 182, 'cyclic_decay': True},  # UserKNN
    {'view_weight': 0.5, 'abs_half_life': 182, 'cyclic_decay': True},  # RecVAE
    {'view_weight': 0.5, 'abs_half_life': 365}  # MultVAE
]

model_dict = {'model_classes': model_classes, 'models_hyp': models_hyp, 'is_content_based': is_content_based, 'URM_params' : URM_params}

# TUNING AND TRAINING
lgb_model, _ = LGB_tune_test(
    train_sessions_df=train_sessions_df.copy(),
    train_purchases_df=train_purchases_df.copy(),
    item_features_df=item_features_df.copy(),
    session_attributes_train_df=session_attributes_train_df.copy(),
    item_attributes_df=item_attributes_df.copy(),
    model_dict=model_dict,
    num_trials=200,
    num_folds=5,
    num_boost_round=1500
)

# INFERENCE
prediction_df = LGB_test_submission(
    train_sessions_df=train_sessions_df.copy(),
    train_purchases_df=train_purchases_df.copy(),
    item_features_df=item_features_df.copy(),
    test_sessions_df=test_sessions_df.copy(),
    candidate_items_df=candidate_items_df.copy(),
    session_attributes_test_df=session_attributes_test_df.copy(),
    item_attributes_df=item_attributes_df.copy(),
    lgb_model=lgb_model,
    model_dict=model_dict,
)

# CHECK IF SUBMISSION_DF IS WELL-CONSTRUCTED
print(all(np.sort(prediction_df['session_id'].unique()) == np.sort(test_sessions_df['session_id'].unique())))
print(all(prediction_df.groupby(['session_id']).count()['item_id'].values == 100))
print(all(np.isin(prediction_df['item_id'].unique(), candidate_items_df['item_id'].unique())))
print(all(prediction_df.groupby(['session_id'])['item_id'].nunique() == 100))
