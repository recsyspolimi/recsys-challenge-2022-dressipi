import os
import random

import numpy as np
from random import sample
import time
from tqdm import tqdm

from DressipiChallenge.GRU4Rec.gru4rec import GRU4Rec
from DressipiChallenge.GRU4Rec.evaluation import *


class GRU4RecRecommender:
    RECOMMENDER_NAME = "GRU4RecRecommender"

    def __init__(self, train_df, test_sessions_df, URM_train,
                 session_key='session_id', item_key='item_id', time_key='date'):

        self.train_df = train_df.copy()
        self.test_sessions_df = test_sessions_df.copy()
        self.URM_dummy = URM_train
        self.n_items = URM_train.shape[-1]
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key

        # remove cold items
        sessions_test = self.test_sessions_df['session_id'].unique()

        items_train = self.train_df['item_id'].unique()
        items_test = self.test_sessions_df['item_id'].unique()
        cold_input_items = items_test[np.isin(items_test, items_train, invert=True)]

        self.test_sessions_df = self.test_sessions_df[
            np.isin(self.test_sessions_df['item_id'].values, cold_input_items, invert=True)]

        # find sessions with only cold items
        sessions_input = self.test_sessions_df['session_id'].unique()  # after removing cold items
        self.empty_sessions = sessions_test[np.isin(sessions_test, sessions_input, invert=True)]

    def get_URM_train(self):
        return self.URM_dummy

    def set_URM_train(self, URM_train):
        return

    def set_items_to_ignore(self, mapped_items_to_ignore):
        # set to ignore not candidates items
        self.items_to_ignore_ids = np.array(mapped_items_to_ignore, dtype=np.int)

    def reset_items_to_ignore(self):
        self.items_to_ignore_ids = np.array([], dtype=np.int)

    def _remove_custom_items_on_scores(self, prediction_scores):
        # not_candidate_idx = np.argwhere(np.isin(self.prediction_items, self.candidate_items, invert=True))
        prediction_scores[:, self.items_to_ignore_ids] = -np.inf  # n_sessions x n_items
        return prediction_scores

    def fit(self,
            loss='bpr-max',
            final_act='elu',
            hidden_act='tanh',
            final_leaky_alpha=0.01,
            hidden_leaky_alpha=0.01,
            final_elu_alpha=1.0,
            hidden_elu_alpha=1.0,
            n_layers=1,
            n_units=100,
            n_epochs=1,
            batch_size=32,
            dropout_p_hidden=0.0,
            dropout_p_embed=0.0,
            learning_rate=0.1,
            momentum=0.0,
            embedding=False,
            n_embedding=0,
            n_sample=2048,
            sample_alpha=0.75,
            constrained_embedding=False,
            bpreg=1.0,
            logq=0.0,
            sigma=0.0,
            init_as_normal=False):

        layers = [n_units // (2 ** p) for p in range(n_layers)]

        if hidden_act == 'elu':
            hidden_act += ('-' + str(hidden_elu_alpha))
        elif hidden_act == 'leaky':
            hidden_act += ('-' + str(hidden_leaky_alpha))

        if final_act == 'elu':
            final_act += ('-' + str(final_elu_alpha))
        elif final_act == 'leaky':
            final_act += ('-' + str(final_leaky_alpha))

        if embedding:
            n_embed = n_embedding
        else:
            n_embed = 0

        self.gru = GRU4Rec(loss=loss,
                           final_act=final_act, hidden_act=hidden_act, layers=layers,
                           n_epochs=n_epochs, batch_size=batch_size,
                           dropout_p_hidden=dropout_p_hidden, dropout_p_embed=dropout_p_embed,
                           learning_rate=learning_rate, momentum=momentum,
                           embedding=n_embed, constrained_embedding=constrained_embedding,
                           n_sample=n_sample, sample_alpha=sample_alpha,
                           bpreg=bpreg, logq=logq,
                           sigma=sigma, init_as_normal=init_as_normal,
                           session_key=self.session_key, item_key=self.item_key, time_key=self.time_key)

        self.gru.fit(self.train_df)

    def predict_next(self, sessions_id_array, input_items_prediction, predict_for_item_ids, batch):

        prediction_df = self.gru.predict_next_batch(sessions_id_array, input_items_prediction,
                                                    predict_for_item_ids=predict_for_item_ids,
                                                    batch=batch).T  # notice the transpose operator

        return prediction_df

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        user_id_array = np.array(user_id_array)

        non_empty_sessions_mask = np.isin(user_id_array, self.empty_sessions, invert=True)
        sessions_id_array = user_id_array[non_empty_sessions_mask]
        sessions_id_array_idx = np.arange(len(user_id_array))[non_empty_sessions_mask]
        self.batch_non_empty_sessions_id = sessions_id_array_idx
        # self.empty_sessions_batch = user_id_array[np.isin(user_id_array, self.empty_sessions)]
        assert len(sessions_id_array) == len(self.batch_non_empty_sessions_id)

        batch_sessions_df = self.test_sessions_df[self.test_sessions_df['session_id'].isin(sessions_id_array)]
        view_sessions_id_arr = batch_sessions_df['session_id'].values
        input_items_prediction = batch_sessions_df['item_id'].values
        assert len(view_sessions_id_arr) == len(input_items_prediction)

        view_sessions_id_arr_cpy = view_sessions_id_arr.copy()
        sessions_len_gby = batch_sessions_df.groupby('session_id').size()
        sessions_len = sessions_len_gby.values
        # session_orig_ids = sessions_len_gby.index
        max_sessions_len = max(sessions_len)

        while max_sessions_len > 0:
            # print(max_sessions_len)
            session_idx = np.argwhere(sessions_len == max_sessions_len)
            view_sessions_id_arr_list = view_sessions_id_arr_cpy.tolist()
            view_idx = [view_sessions_id_arr_list.index(sessions_id_array[k]) for k in session_idx]
            x = view_sessions_id_arr_cpy[view_idx]
            y = input_items_prediction[view_idx]
            view_sessions_id_arr_cpy[view_idx] = -1
            sessions_len[sessions_len == max_sessions_len] -= 1
            max_sessions_len -= 1
            assert len(x) == len(y)
            # return scores of all candidate_items for every event of the batch (view_sessions_id_arr)
            prediction_df = self.predict_next(x, y,
                                              predict_for_item_ids=None,
                                              batch=len(x))

        # items for which a score was computed
        self.prediction_items = np.array(prediction_df.columns).astype(int)
        # prediction scores of batch sessions in a multi-array
        prediction_scores = prediction_df.loc[:].values

        assert len(self.prediction_items) == prediction_scores.shape[-1]

        tmp_prediction_scores = np.zeros((len(user_id_array), prediction_scores.shape[-1]))
        tmp_prediction_scores[sessions_id_array_idx, :] = prediction_scores

        assert len(tmp_prediction_scores) == len(user_id_array)

        final_prediction_scores = np.zeros((len(user_id_array), self.n_items))
        # final_prediction_scores[:] = -np.inf
        final_prediction_scores[:, self.prediction_items] = tmp_prediction_scores

        return final_prediction_scores

    def recommend(self, user_id_array, cutoff=100, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):

        user_id_array = np.array(user_id_array)

        prediction_scores = self._compute_item_score(user_id_array)  # n_sessions x n_items
        #print('End compute_item_score')

        sessions_id_array = user_id_array[self.batch_non_empty_sessions_id]

        # remove non candidates
        prediction_scores = self._remove_custom_items_on_scores(prediction_scores)
        #print('End remove candidates')

        # remove already seen items
        already_seen_items_idx = [
            self.test_sessions_df[self.test_sessions_df['session_id'] == session_id]['item_id'].unique()
            if session_id in sessions_id_array
            else np.array([])
            for session_id in user_id_array]

        # print('End remove already seen idx')
        for i in range(len(prediction_scores)):
            seen_idx = already_seen_items_idx[i]
            if len(seen_idx) > 0:
                prediction_scores[i, seen_idx] = -np.inf  # n_sessions x n_items
        #print('End remove already seen loop')

        # ranking of items for every session
        # ranking_lists = [[item for _, item in
        #                  sorted(zip(single_prediction_scores, self.prediction_items), reverse=True)][:cutoff]
        #                 for single_prediction_scores in prediction_scores]  # n_sessions x cutoff
        rankings_idxs = (-prediction_scores).argpartition(cutoff)[:, :cutoff]
        sorted_rankings_idxs = np.array([None] * prediction_scores.shape[0])
        for i in range(prediction_scores.shape[0]):
            sorted_rankings_idxs[i] = rankings_idxs[i, np.argsort(-prediction_scores[i, rankings_idxs[i]])]

        assert len(sorted_rankings_idxs) == len(user_id_array)
        ranking_lists = [rid
                         if i in self.batch_non_empty_sessions_id
                         else random.sample(self.prediction_items[
                                                np.isin(self.prediction_items,
                                                        self.items_to_ignore_ids,
                                                        invert=True)].tolist(), cutoff)
                         for i, rid in enumerate(sorted_rankings_idxs)]
        #print('End ranking')

        print('n_items: {}'.format(self.n_items))
        print('ranking_lists shape: {}'.format(np.shape(ranking_lists)))
        print('prediction_scores shape: {}'.format(np.shape(prediction_scores)))

        if return_scores:
            return ranking_lists, prediction_scores

        return ranking_lists

    def save_model(self, folder_path, file_name=None):
        self.gru.savemodel('./'+folder_path+'/'+file_name+'.pkl')
        print("Saving complete")

    def load_model(self, folder_path, file_name=None):
        print("Try to load")
        self.gru = GRU4Rec.loadmodel('./'+folder_path+'/'+file_name+'.pkl')
        print("Loading complete")
