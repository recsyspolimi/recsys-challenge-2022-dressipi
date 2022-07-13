import os
from logging import getLogger

import numpy as np

import pandas as pd
from recbole.quick_start import load_data_and_model
from tqdm import tqdm

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, Interaction
from recbole.model.sequential_recommender.bert4rec import *
from recbole.trainer import Trainer
from recbole.utils import init_logger, set_color

from DressipiChallenge.Recommenders.BaseRecommender import BaseRecommender
from DressipiChallenge.Recommenders.NonPersonalizedRecommender import TopPop


def add_zero_padding(item_list, max_list_len=100):
    zero_padding = [0 for _ in range(max_list_len - len(item_list))]
    item_list.extend(zero_padding)


def token2id(item_list, dataset):
    item_list = np.array(item_list)
    return list(dataset.token2id(dataset.iid_field, item_list.astype(str)))


def token2id_single(item, dataset):
    return dataset.token2id(dataset.iid_field, str(item))


OPTIONS = {

    # general
    'gpu_id': 0,
    'use_gpu': True,
    'seed': 2020,
    'state': 'INFO',
    'reproducibility': True,
    'data_path': './RecBole_Dataset/BERT4Rec/',
    'dataset': 'dressipi',
    'checkpoint_dir': 'RecBole_Ckpt/',
    'show_progress': True,
    'save_dataset': False,
    'dataset_save_path': None,
    'save_dataloaders': False,
    'dataloaders_save_path': None,
    'log_wandb': False,
    # wandb_project: 'recbole'

    # evaluation settings
    'eval_args': {
        'split': {'RS': [1, 0, 0]},
        'group_by': None,
        'order': 'TO',
        'mode': 'full'
    },
    'repeatable': True,
    'metrics': ["Recall", "MRR", "NDCG", "Hit", "Precision"],
    'topk': [100],
    'valid_metric': 'MRR@100',
    'valid_metric_bigger': True,
    'eval_batch_size': 512,
    'metric_decimal_place': 4,

    # Atomic File Format
    'field_separator': "\t",
    'seq_separator': " ",

    # Common Features
    'USER_ID_FIELD': 'session_id',
    'ITEM_ID_FIELD': 'item_id',
    'RATING_FIELD': None,
    'TIME_FIELD': 'timestamp',
    'seq_len': None,
    # Label for Point-wise DataLoader
    'LABEL_FIELD': 'label',
    # NegSample Prefix for Pair-wise DataLoader
    'NEG_PREFIX': 'neg_',
    # Sequential Model Needed
    'ITEM_LIST_LENGTH_FIELD': 'item_length',
    'LIST_SUFFIX': '_list',
    'MAX_ITEM_LIST_LENGTH': 100,
    'POSITION_FIELD': 'position_id',
    # Knowledge-based Model Needed
    'HEAD_ENTITY_ID_FIELD': 'head_id',
    'TAIL_ENTITY_ID_FIELD': 'tail_id',
    'RELATION_ID_FIELD': 'relation_id',
    'ENTITY_ID_FIELD': 'entity_id',

    # Selectively Loading
    'load_col': {
        'inter': ['session_id', 'item_id', 'timestamp']
    },
    'unload_col': None,
    'unused_col': None,

    # Filtering
    'rm_dup_inter': None,
    'val_interval': None,
    'filter_inter_by_user_or_item': True,
    'user_inter_num_interval': None,
    'item_inter_num_interval': None,

    # Preprocessing
    'alias_of_user_id': None,
    'alias_of_item_id': None,
    'alias_of_entity_id': None,
    'alias_of_relation_id': None,
    'preload_weight': None,
    'normalize_field': None,
    'normalize_all': True
}


class BERT4RecRecommender(BaseRecommender):
    RECOMMENDER_NAME = "BERT4RecRecommender"

    """
    Attributes:
    
    - train_set_df: pd.DataFrame containing all training views and purchases
        cols = ['session_id', 'item_id', 'date']
    - temp_test_set_df: temporary test_set_df before its construction in fit(...)
    - test_set_df: pd.DataFrame containing that contains all test views
        cols = ['session_id', 'item_id', 'date']
    
    - dataset: recbole internal representation full dataset
    - train_data: recbole internal representation training data
    - valid_data: recbole internal representation training data
    
    - items_to_ignore_ID: items that are not candidates
    - candidate_item_id_array: items to recommend
    - train_item_id_array: all items that appeared in train_set_df
    - all_item_id_array: all items that appeared in train_set_df and test_set_df
    
    - model: recbole.BERT4Rec model
    
    - verbose: bool that indicates whether to show or not progress bar during training
    - config: recbole internal representation configuration file
    - logger: object to show logs
    """

    def __init__(self,
                 train_df,
                 test_sessions_df,
                 URM_train,
                 verbose=True):
        super().__init__(URM_train)

        top_pop_recommender = TopPop(URM_train)
        top_pop_recommender.fit()
        self.top_pop_item_id_array = np.array(top_pop_recommender.recommend(0))

        self.n_items = URM_train.shape[-1]
        self.verbose = verbose

        self.all_item_id_array = np.arange(self.n_items)
        self.recommendable_item_id_array = train_df['item_id'].unique()
        self.candidate_item_id_array = self.all_item_id_array

        assert all(np.isin(self.recommendable_item_id_array, self.all_item_id_array)) \
               and all(np.isin(test_sessions_df['item_id'].unique(), self.all_item_id_array))

        self.train_set_df = self._build_train_set_df(train_df[['session_id', 'item_id', 'date']])
        self.temp_test_views_df = test_sessions_df.copy()

        self.test_views_df = None

        dataset_name = OPTIONS['dataset']
        dataset_dir = OPTIONS['data_path'] + dataset_name + '/'
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        self.train_set_df.to_csv(dataset_dir + dataset_name + '.inter', sep='\t', index=False)

        self.config = None
        self.model = None
        self.dataset = None
        self.train_data = None
        self.logger = None
        self.trainer = None

    def set_items_to_ignore(self, items_to_ignore):
        self.items_to_ignore_flag = True
        self.items_to_ignore_ID = np.array(items_to_ignore, dtype=np.int)
        candidate_item_id_mask = ~np.isin(self.all_item_id_array, self.items_to_ignore_ID)
        self.candidate_item_id_array = self.all_item_id_array[candidate_item_id_mask]

    def set_URM_train(self, URM_train_new, **kwargs):
        self.URM_train = URM_train_new

    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        saved_model_file = folder_path + file_name + '.pth'
        self._print("Saving model in file '{}'".format(saved_model_file))
        state = {
            'config': self.trainer.config,
            'epoch': self.config['epochs'],
            'cur_step': self.trainer.cur_step,
            'best_valid_score': self.trainer.best_valid_score,
            'state_dict': self.trainer.model.state_dict(),
            'other_parameter': self.trainer.model.other_parameter(),
            'optimizer': self.trainer.optimizer.state_dict(),
        }
        torch.save(state, saved_model_file)
        if self.verbose:
            self.logger.info(set_color('Saving current', 'blue') + f': {saved_model_file}...')
        self._print("Saving complete")

    def load_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        model_file = folder_path + file_name + '.pth'
        self._print("Loading model from file '{}'".format(model_file))
        self.config, self.model, self.dataset, self.train_data, _, _ = load_data_and_model(model_file=model_file)
        init_logger(self.config)
        self.logger = getLogger()
        self.logger.info(self.model)
        self.test_views_df = self._build_test_views_df()
        self._print("Loading complete")

    def fit(self,
            eval_args=None,
            n_layers=5,
            n_heads=2,
            hidden_size=32,
            inner_size=8,
            hidden_dropout_prob=0.4,
            attn_dropout_prob=0.4,
            hidden_act='gelu',
            layer_norm_eps=1e-10,
            mask_ratio=0.3,
            initializer_range=1e-4,
            learning_rate=1e-3,
            learner='adam',
            epochs=30,
            train_batch_size=512,
            neg_sampling=None,
            eval_step=1,
            stopping_step=5,
            clip_grad_norm=None,
            weight_decay=0.00001,
            loss_decimal_place=4,
            require_pow=False,
            loss_type='BPR'):

        if neg_sampling is None and loss_type == 'BPR':
            neg_sampling = {'uniform': 1}

        if eval_args is None:
            eval_args = {
                'split': {'RS': [1, 0, 0]},
                'group_by': None,
                'order': 'TO',
                'mode': 'full'
            }

        model_hp = {

            # model settings
            'n_layers': n_layers,
            'n_heads': n_heads,
            'hidden_size': hidden_size,
            'inner_size': inner_size,
            'hidden_dropout_prob': hidden_dropout_prob,
            'attn_dropout_prob': attn_dropout_prob,
            'hidden_act': hidden_act,
            'layer_norm_eps': layer_norm_eps,
            'mask_ratio': mask_ratio,
            'initializer_range': initializer_range,
            'learning_rate': learning_rate,
            'learner': learner,

            # training settings
            'epochs': epochs,
            'train_batch_size': train_batch_size,
            'neg_sampling': neg_sampling,
            'eval_step': eval_step,
            'stopping_step': stopping_step,
            'clip_grad_norm': clip_grad_norm,
            'weight_decay': weight_decay,
            'loss_decimal_place': loss_decimal_place,
            'require_pow': require_pow,
            'loss_type': loss_type,

            # evaluation settings
            'eval_args': eval_args
        }

        config_dict = OPTIONS
        config_dict.update(model_hp)

        self.config = Config(model='BERT4Rec', dataset='dressipi', config_dict=config_dict)
        self.dataset = create_dataset(self.config)

        init_logger(self.config)
        self.logger = getLogger()
        self.logger.info(self.config)
        self.logger.info(self.dataset)

        self.test_views_df = self._build_test_views_df()

        self.train_data, _, _ = data_preparation(self.config, self.dataset)

        self.model = BERT4Rec(self.config, self.train_data.dataset).to(self.config['device'])
        self.logger.info(self.model)

        self.trainer = Trainer(self.config, self.model)
        _, _ = self.trainer.fit(self.train_data, show_progress=self.verbose)

    def recommend(self, user_id_array,
                  cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False,
                  top_pop_on_non_recommendable=False):

        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False
            if not isinstance(user_id_array, type(np.array([]))):
                user_id_array = np.array(user_id_array)

        if cutoff is None:
            cutoff = len(self.all_item_id_array) - 1
        cutoff = min(cutoff, len(self.all_item_id_array) - 1)

        scores_batch = self._compute_item_score(user_id_array)

        recommendable_user_id_mask = np.isin(user_id_array, self.test_views_df['session_id'].values)
        recommendable_user_id_array = user_id_array[recommendable_user_id_mask]

        if remove_seen_flag:
            for batch_idx in range(len(user_id_array)):
                user_id = user_id_array[batch_idx]
                if np.isin(user_id, self.test_views_df['session_id'].values):
                    already_seen_item_id_array = self.temp_test_views_df[
                        self.temp_test_views_df['session_id'] == user_id]['item_id'].unique()
                    scores_batch[batch_idx, already_seen_item_id_array] = -np.inf

        if remove_custom_items_flag:
            scores_batch[:, self.items_to_ignore_ID] = -np.inf

        noninf_item_ids_batch = np.where(~np.isinf(scores_batch))[1]

        # DEBUGGING
        assert all(np.isin(self.candidate_item_id_array, noninf_item_ids_batch))

        relevant_items_batch = (-scores_batch).argpartition(cutoff, axis=1)[:, :cutoff]
        rankings = np.array([None] * scores_batch.shape[0])
        for i in range(len(scores_batch)):
            rankings[i] = relevant_items_batch[i, np.argsort(-scores_batch[i, relevant_items_batch[i]])]

        rankings_list = [[]] * scores_batch.shape[0]

        # Remove from the recommendation list any item that has a -inf score
        # Since -inf is a flag to indicate an item to remove
        for user_index in range(len(user_id_array)):
            if user_index in recommendable_user_id_array:
                user_recommendation_list = rankings[user_index]
                user_item_scores = scores_batch[user_index, user_recommendation_list]
                not_inf_scores_mask = ~np.isinf(user_item_scores)
                user_recommendation_list = user_recommendation_list[not_inf_scores_mask]
                rankings_list[user_index] = list(user_recommendation_list)
            else:
                if top_pop_on_non_recommendable:
                    rankings_list[user_index] = self.top_pop_item_id_array[:cutoff]
                else:
                    rankings_list[user_index] = random.sample(list(self.candidate_item_id_array), cutoff)

        if single_user:
            rankings_list = rankings_list[0]

        if return_scores:
            return rankings_list, scores_batch
        else:
            return rankings_list

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
        else:
            if not isinstance(user_id_array, type(np.array([]))):
                user_id_array = np.array(user_id_array)

        recommendable_user_id_mask = np.isin(user_id_array, self.test_views_df['session_id'].values)
        recommendable_user_id_array = user_id_array[recommendable_user_id_mask]

        interactions = Interaction(self.test_views_df[self.test_views_df['session_id'].isin(user_id_array)])
        interactions = interactions.to(torch.device('cuda:0'))

        bert_recommendable_scores_batch_internal_ids = self.model.full_sort_predict(interactions)
        # [1:] because first item is score of '[PAD]' (-inf)
        bert_recommendable_scores_batch_internal_ids = np.array([
            elem.cpu().detach().numpy()[1:] for elem in bert_recommendable_scores_batch_internal_ids])

        inverse_mapping = self.dataset.id2token(
            self.dataset.iid_field,
            np.arange(1, bert_recommendable_scores_batch_internal_ids.shape[1] + 1)
        ).astype(int)

        bert_scores_batch = np.zeros((recommendable_user_id_array.shape[0], self.all_item_id_array.shape[0]))
        bert_scores_batch[:, inverse_mapping] = bert_recommendable_scores_batch_internal_ids

        scores_batch = np.zeros((user_id_array.shape[0], self.all_item_id_array.shape[0]))
        scores_batch[recommendable_user_id_mask] = bert_scores_batch

        # DEBUGGING
        nonzero_item_ids = np.where(scores_batch != 0.0)[1]
        assert all(np.isin(nonzero_item_ids, self.recommendable_item_id_array))

        return scores_batch

    def _build_train_set_df(self, train_set_df_old):
        train_set_df = train_set_df_old.copy()
        zero_time = pd.to_datetime('2020-01-01 00:00:00.000')
        train_set_df.sort_values(['session_id', 'date'], inplace=True)
        train_set_df.reset_index(drop=True, inplace=True)
        train_set_df.rename(columns={"session_id": "session_id:token",
                                     "item_id": "item_id:token",
                                     "date": "timestamp:float"}, inplace=True)
        train_set_df['timestamp:float'] = pd.to_datetime(train_set_df['timestamp:float'],
                                                         format="%Y-%m-%d %H:%M:%S.%f")
        train_set_df['timestamp:float'] = (train_set_df['timestamp:float'] - zero_time
                                           ).dt.total_seconds().astype(int)
        return train_set_df

    def _build_test_views_df(self):
        bert_test_sessions_df = self.temp_test_views_df[
            self.temp_test_views_df['item_id'].isin(self.recommendable_item_id_array)]
        test_views_df = pd.DataFrame()
        test_views_df['session_id'] = bert_test_sessions_df['session_id'].unique()
        # each element of 'item_id_list' column is a list [view1, view2, ...]
        test_views_df['item_id_list'] = bert_test_sessions_df.groupby('session_id')['item_id'].apply(list).values
        test_views_df['item_id_list'] = test_views_df['item_id_list'].apply(token2id, dataset=self.dataset).values
        test_views_df['item_length'] = test_views_df['item_id_list'].apply(len).values
        # we add padding to have lists of the same length [view1, view2, ..., viewn, 0, 0, ..., 0]
        test_views_df['item_id_list'].apply(add_zero_padding, max_list_len=self.config['MAX_ITEM_LIST_LENGTH'])
        return test_views_df
