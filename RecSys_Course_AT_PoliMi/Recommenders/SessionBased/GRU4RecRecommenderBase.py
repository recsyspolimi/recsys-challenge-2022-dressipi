import numpy as np


class GRU4RecRecommenderBase:
    # Base class for a GRU4RecRecommender in order to fit it to the original environment

    RECOMMENDER_NAME = "GRU4RecRecommenderBase"

    def __init__(self, train_df, test_sessions_df, candidate_items, URM_train_dummy, only_last=False,
                 session_key='session_id', item_key='item_id', time_key='date'):

        self.train_df = train_df
        self.test_sessions_df = test_sessions_df
        self.candidate_items = candidate_items
        self.URM_dummy = URM_train_dummy
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key

        if only_last:
            self.input_df = self.test_sessions_df.drop_duplicates(['session_id'], keep='last')
        else:
            self.input_df = self.test_sessions_df.copy()

    def get_URM_train(self):
        return self.URM_dummy

    def fit(self):
        raise NotImplementedError

    def recommend(self):
        raise NotImplementedError


class GRU4RecRecommenderNextBase(GRU4RecRecommenderBase):
    # Base class for a GRU4RecRecommender based on next item prediction

    RECOMMENDER_NAME = "GRU4RecRecommenderNextBase"

    def __init__(self, *args, **kwargs):
        super(GRU4RecRecommenderNextBase, self).__init__(*args, **kwargs)

    def predict_next(self, sessions_id_array, input_items_prediction, predict_for_item_ids, batch):
        raise NotImplementedError

    def _compute_item_score(self, sessions_id_array, items_to_compute=None):

        # item_scores = np.empty((len(sessions_id_array), 23691))  # n_items : 23691
        # item_scores[:] = np.nan

        batch_sessions_df = self.input_df[self.input_df['session_id'].isin(sessions_id_array)]
        view_sessions_id_arr = batch_sessions_df['session_id'].values
        input_items_prediction = batch_sessions_df['item_id'].values
        assert len(view_sessions_id_arr) == len(input_items_prediction)

        view_sessions_id_arr_cpy = view_sessions_id_arr.copy()
        sessions_len_gby = batch_sessions_df.groupby('session_id').size()
        sessions_len = sessions_len_gby.values
        #session_orig_ids = sessions_len_gby.index
        max_sessions_len = max(sessions_len)

        while max_sessions_len > 0:
            #print(max_sessions_len)
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

        # keep only scores for last prediction event
        #prediction_df['session_id'] = view_sessions_id_arr
        #prediction_df.drop_duplicates(['session_id'], keep='last', inplace=True)
        #prediction_df.drop(['session_id'], axis=1, inplace=True)

        # items for which a score was computed
        self.prediction_items = np.array(prediction_df.columns).astype(int)
        # prediction scores of batch sessions in a multi-array
        prediction_scores = prediction_df.loc[:].values

        return prediction_scores

    def recommend(self, sessions_id_array, cutoff=100, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):

        prediction_scores = self._compute_item_score(sessions_id_array)  # n_sessions x n_items
        print('End compute_item_score')

        '''
        # sorted items and scores for every session
        sorted_prediction_scores_items = np.array(
            [[score_item for score_item in
              sorted(zip(single_session_prediction_scores, prediction_items), reverse=True)]
             for single_session_prediction_scores in prediction_scores])  # n_sessions x n_candidates x 2

        # final lists of prediction and scores at cutoff 100 removing already-seen items
        final_lists = np.array([single_session_prediction_scores_items[
                                    np.isin(single_session_prediction_scores_items[:, -1],
                                            self.test_sessions_df[self.test_sessions_df['session_id'] == session_id][
                                                'item_id'].unique(),
                                            invert=True)][:cutoff].tolist()
                                for session_id, single_session_prediction_scores_items in
                                zip(sessions_id_array, sorted_prediction_scores_items)])  # n_sessions x cutoff x 2

        final_prediction_lists = final_lists[:, :, -1].astype(int).tolist()
        final_scores = final_lists[:, :, 0]
        '''

        # remove non candidates
        not_candidate_idx = np.argwhere(np.isin(self.prediction_items, self.candidate_items, invert=True))
        prediction_scores[:, not_candidate_idx] = -np.inf  # n_sessions x n_items
        print('End remove candidates')


        # remove already seen items
        already_seen_items_idx = [np.argwhere(np.isin(self.prediction_items,
                                                      self.test_sessions_df[
                                                          self.test_sessions_df['session_id'] == session_id][
                                                          'item_id'].unique(),
                                                      )).ravel() for session_id in sessions_id_array]
        #print('End remove already seen idx')
        for i in range(len(prediction_scores)):
            prediction_scores[i, already_seen_items_idx[i]] = -np.inf  # n_sessions x n_items
        print('End remove already seen loop')


        # ranking of items for every session
        #ranking_lists = [[item for _, item in
        #                  sorted(zip(single_prediction_scores, self.prediction_items), reverse=True)][:100]
        #                 for single_prediction_scores in prediction_scores]  # n_sessions x cutoff
        ranking_idx = (-prediction_scores).argpartition(100)[:, :100]
        ranking_lists = [self.prediction_items[rid] for rid in ranking_idx]
        print('End ranking')

        if return_scores:
            return ranking_lists, prediction_scores

        return ranking_lists


class GRU4RecRecommenderWholeBase(GRU4RecRecommenderBase):
    # Base class for a GRU4RecRecommender based on last session item prediction

    RECOMMENDER_NAME = "GRU4RecRecommenderWholeBase"

    def __init__(self, *args, **kwargs):
        super(GRU4RecRecommenderWholeBase, self).__init__(*args, **kwargs)

    def recommend(self, sessions_id_array, cutoff=100, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):

        # extract item views
        items_view = self.test_sessions_df.groupby(['session_id']).agg(list)['item_id'].values
        # add padding
        session_max_length = max(self.test_sessions_df.groupby('session_id').size())
        input_items_prediction = np.ones((len(sessions_id_array), session_max_length), dtype=int) * 23691
        for i in range(len(sessions_id_array)):
            input_items_prediction[i, -len(items_view[i]):] = items_view[i]

        # input_items_prediction.shape : sessions_id_array_length x session_max_length (TF timesteps)
        assert len(sessions_id_array) == len(input_items_prediction)

        prediction_df = self.predict_next(sessions_id_array, input_items_prediction,
                                          predict_for_item_ids=self.candidate_items,
                                          batch=len(sessions_id_array))
        # items for which a score was computed
        prediction_items = prediction_df.columns
        # prediction scores of batch sessions in a multi-array
        prediction_scores = prediction_df.loc[:].values

        # sorted items for every session
        sorted_prediction_scores_items = np.array(
            [[score_item for score_item in
              sorted(zip(single_session_prediction_scores, prediction_items), reverse=True)]
             for single_session_prediction_scores in prediction_scores])  # n_sessions x n_candidates

        # final list of prediction at cutoff 100 removing already-seen items
        final_lists = np.array([single_session_prediction_scores_items[
                                    np.isin(single_session_prediction_scores_items[:, -1],
                                            self.test_sessions_df[self.test_sessions_df['session_id'] == session_id][
                                                'item_id'].unique(),
                                            invert=True)][:cutoff].tolist()
                                for session_id, single_session_prediction_scores_items in
                                zip(sessions_id_array, sorted_prediction_scores_items)])  # n_sessions x cutoff

        final_prediction_lists = final_lists[:, :, -1].astype(int).tolist()
        final_score_lists = final_lists[:, :, 0].tolist()

        if return_scores:
            return final_prediction_lists, final_score_lists

        return final_prediction_lists
