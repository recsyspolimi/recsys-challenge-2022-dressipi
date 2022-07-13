import os
import sys

sys.path.append('/Recommenders')
sys.path.append('../')

from skopt.space import Real, Integer, Categorical

import pandas as pd
import numpy as np

from Pipeline.data_extraction import get_dataframes
from Pipeline.matrices_creation import create_URM_GRU4Rec
from Pipeline.utils import create_candidate_set, create_mapping
from Pipeline.data_splitting import *

from Evaluation.Evaluator import EvaluatorHoldout
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Recommenders.DataIO import DataIO
from Recommenders.SessionBased.GRU4RecRecommender import GRU4RecRecommender

############ DATA PREPARATION ############

item_features_df, train_sessions_df, train_purchases_df, _, _ = get_dataframes()


train_sessions_df.sort_values(['session_id', 'date'], inplace=True)
train_purchases_df.sort_values(['session_id'], inplace=True)

'''
# Dataframes cut
train_intervals_list = [('2020-01-01', '2021-06-01')]
train_sessions_df = keep_sessions_in_intervals(train_sessions_df, train_intervals_list)
train_purchases_df = align_purchases_to_views(train_sessions_df, train_purchases_df)

# Dataframes reindexing
session_mapping = create_mapping(train_sessions_df['session_id'])
item_mapping = create_mapping(item_features_df['item_id'])

train_sessions_df['session_id'] = train_sessions_df['session_id'].map(session_mapping)
train_sessions_df['item_id'] = train_sessions_df['item_id'].map(item_mapping)
train_purchases_df['session_id'] = train_purchases_df['session_id'].map(session_mapping)
train_purchases_df['item_id'] = train_purchases_df['item_id'].map(item_mapping)

# Build final train dataframe
train_set_df = pd.concat([train_sessions_df, train_purchases_df]).sort_values(['session_id', 'date'])
final_train_intervals_list = [('2020-01-01', '2021-05-01')]  # exclude last month
train_set_df = keep_sessions_in_intervals(train_set_df, final_train_intervals_list)

# Prepare validation dataframe
val_intervals_list = [('2021-05-01', '2021-06-01')]
val_sessions_df = keep_sessions_in_intervals(train_sessions_df, val_intervals_list)
val_purchases_df = align_purchases_to_views(val_sessions_df, train_purchases_df)

# Items
train_items = train_set_df['item_id'].unique()
test_items = val_sessions_df['item_id'].unique()
cold_test_items = test_items[np.isin(test_items, train_items, invert=True)]

# Final validation dataframe
val_sessions_df = val_sessions_df[np.isin(val_sessions_df['item_id'].values, cold_test_items, invert=True)]
val_set_df = align_purchases_to_views(val_sessions_df, val_purchases_df)
# last_view_df = val_sessions_df.drop_duplicates(['session_id'], keep='last')

cold_sessions = val_purchases_df[np.isin(val_purchases_df['session_id'].values,
                                         val_set_df['session_id'].unique(), invert=True)]['session_id'].unique()

# Create URMs
URM_val = create_URM_GRU4Rec(val_set_df, session_mapping, item_mapping)
URM_dummy = create_URM_GRU4Rec(val_sessions_df, session_mapping, item_mapping)

# Params for recommendation
session_ids_arr = val_sessions_df['session_id'].values  # not used here
input_item_ids_arr = val_sessions_df['item_id'].values  # not used here
candidate_items = create_candidate_set(val_set_df)  # already mapped
candidate_items = candidate_items[np.isin(candidate_items, train_items)]
'''

tot_df = pd.concat([train_sessions_df.copy(), train_purchases_df.copy()])

# cut item features df
tot_items = tot_df['item_id'].unique()
item_features_df = item_features_df[item_features_df['item_id'].isin(tot_items)]

# cut tmp dfs
final_train_intervals_list = [('2020-01-01', '2021-05-01')]  # exclude last month
train_set_df = keep_sessions_in_intervals(tot_df, final_train_intervals_list)
final_view_intervals_list = [('2021-05-01', '2021-06-01')]  # exclude last month
val_sessions_df = keep_sessions_in_intervals(train_sessions_df.copy(), final_view_intervals_list)

# find cold items
train_items = train_set_df['item_id'].unique()
test_items = val_sessions_df['item_id'].unique()
cold_test_items = test_items[np.isin(test_items, train_items, invert=True)]

# remove cold sessions from dfs
cut_val_sessions_df = val_sessions_df[~val_sessions_df['item_id'].isin(cold_test_items)]
orig_val_sessions = val_sessions_df['session_id'].unique()
cut_val_sessions = cut_val_sessions_df['session_id'].unique()
cold_sessions = orig_val_sessions[np.isin(orig_val_sessions, cut_val_sessions, invert=True)]

train_sessions_df = train_sessions_df[~train_sessions_df['session_id'].isin(cold_sessions)]
train_purchases_df = align_purchases_to_views(train_sessions_df, train_purchases_df)

# now df have already the cold sessions removed
val_start_ts = '2021-05-01'
val_end_ts = '2021-06-01'

train_set_df, val_purchases_df, val_views_df, train_session_mapping, val_session_mapping, item_mapping, \
mapped_items_to_ignore_val, val_sessions_arr = split_dataframes_val(
    train_sessions_df.copy(), train_purchases_df.copy(), item_features_df.copy(),
    val_start_ts='2021-05-01', val_end_ts='2021-06-01', unique_interactions=False)

URM_val = create_URM_GRU4Rec(val_purchases_df, val_session_mapping, item_mapping)
URM_dummy = create_URM_GRU4Rec(train_set_df, train_session_mapping, item_mapping)

############ SEARCH ############

hp_range_dictionary = {
    'loss': Categorical(['bpr-max']),  # NO cross-entropy!!! ** ['bpr-max', 'bpr', 'top1-max', 'top1']
    'final_act': Categorical(['linear', 'leaky', 'elu']),
    'hidden_act': Categorical(['linear', 'relu', 'tanh', 'leaky', 'elu']),
    'final_leaky_alpha': Real(low=1e-2, high=0.25, prior='log-uniform'),
    'hidden_leaky_alpha': Real(low=1e-3, high=1e-1, prior='log-uniform'),
    'final_elu_alpha': Real(low=0.8, high=2.0, prior='uniform'),
    'hidden_elu_alpha': Real(low=0.1, high=1.0, prior='uniform'),
    'n_units': Integer(50, 200),
    'n_epochs': Integer(5, 12),
    'batch_size': Categorical([32]),
    'dropout_p_hidden': Real(low=0.0, high=0.2, prior='uniform'),
    'dropout_p_embed': Real(low=0.0, high=0.2, prior='uniform'),
    'learning_rate': Real(low=1e-3, high=1e-1, prior='log-uniform'),
    'momentum': Real(low=0.0, high=0.3, prior='uniform'),
    'embedding': Categorical([True, False]),
    'n_embedding': Integer(10, 200),
    'n_sample': Categorical([1024, 2048]),  # ** 4096 might raise NaN error
    'sample_alpha': Real(low=0.0, high=0.8, prior='uniform'),
    'constrained_embedding': Categorical([True, False]),
    'bpreg': Real(low=1e-3, high=1.0, prior='log-uniform'),
    'logq': Real(low=0.0, high=1.0, prior='uniform'),
    # 'sigma': Real(low=0.0, high=10.0, prior='uniform'),
    'init_as_normal': Categorical([False]),
}

evaluator_val = EvaluatorHoldout(URM_val, cutoff_list=[100], ignore_items=mapped_items_to_ignore_val)

recommender_class = GRU4RecRecommender

hp_search = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_val)

recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[train_set_df, val_views_df, URM_dummy],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={}
)

output_folder_path = "result_experiments/"
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

print("Parameters set, starts optimization...")

hp_search.search(recommender_input_args,
                 hyperparameter_search_space=hp_range_dictionary,
                 n_cases=45,
                 n_random_starts=15,
                 resume_from_saved=False,
                 save_model="no",
                 output_folder_path=output_folder_path,
                 output_file_name_root=recommender_class.RECOMMENDER_NAME + "_correct_batch",
                 metric_to_optimize="MRR",
                 cutoff_to_optimize=100)

data_loader = DataIO(folder_path=output_folder_path)
search_metadata = data_loader.load_data(recommender_class.RECOMMENDER_NAME + "_metadata.zip")

hyper_parameters_df = search_metadata["hyperparameters_df"]
print(hyper_parameters_df)

result_on_validation_df = search_metadata["result_on_validation_df"]
print(result_on_validation_df)

best_hyper_parameters = search_metadata["hyperparameters_best"]
print(best_hyper_parameters)
