#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from DressipiChallenge.Recommenders.Recommender_utils import check_matrix
from DressipiChallenge.Recommenders.BaseSimilarityMatrixRecommender import BaseUserSimilarityMatrixRecommender

from DressipiChallenge.Recommenders.IR_feature_weighting import okapi_BM_25, TF_IDF
import numpy as np
import scipy.sparse as sps
from DressipiChallenge.Utils.set_zeros import set_zeros_row_col

from DressipiChallenge.Recommenders.Similarity.Compute_Similarity import Compute_Similarity


class UserKNNCFRecommender(BaseUserSimilarityMatrixRecommender):
    """ UserKNN recommender"""

    RECOMMENDER_NAME = "UserKNNCFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]


    def __init__(self, URM_train, verbose = True):
        super(UserKNNCFRecommender, self).__init__(URM_train, verbose = verbose)


    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting = "none", URM_bias = False, **similarity_args):

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(self.FEATURE_WEIGHTING_VALUES, feature_weighting))

        if URM_bias is not None:
            self.URM_train.data += URM_bias

        if feature_weighting == "BM25":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = okapi_BM_25(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        elif feature_weighting == "TF-IDF":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = TF_IDF(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        similarity = Compute_Similarity(self.URM_train.T, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)

        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')

class UserKNNCFRecommenderStackedXGBoost(BaseUserSimilarityMatrixRecommender):
    """ UserKNN recommender"""

    RECOMMENDER_NAME = "UserKNNCFRecommenderStackedXGBoost"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]


    def __init__(self, URM_train, URM_validation, verbose = True):
        self.URM_train_real = URM_train
        self.URM_validation = URM_validation
        self.users_validation = URM_validation.shape[0]
        self.users_train = URM_train.shape[0]
        super(UserKNNCFRecommenderStackedXGBoost, self).__init__(sps.vstack([URM_train, URM_validation]), verbose = verbose)

    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting = "none", URM_bias = False, **similarity_args):

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(self.FEATURE_WEIGHTING_VALUES, feature_weighting))

        if URM_bias is not None:
            self.URM_train.data += URM_bias

        if feature_weighting == "BM25":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = okapi_BM_25(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        elif feature_weighting == "TF-IDF":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = TF_IDF(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        similarity = Compute_Similarity(self.URM_train.T, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)

        self.W_sparse = similarity.compute_similarity(start_col=self.users_train, end_col=self.users_train+self.users_validation)
        r = np.arange(self.users_train, self.users_train+self.users_validation)
        set_zeros_row_col(self.W_sparse,r,r)
        self.W_sparse = self.W_sparse.transpose()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')

    def set_URM_train(self, URM_train_new, **kwargs):
        pass

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        if any(i < self.users_train for i in user_id_array):
            user_id_array_new = [x+self.users_train for x in user_id_array]
        else:
            user_id_array_new = user_id_array
        return super()._compute_item_score(user_id_array_new, items_to_compute)

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None, remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):
        user_id_array_new = [x+self.users_train for x in user_id_array]
        return super().recommend(user_id_array_new, cutoff, remove_seen_flag, items_to_compute, remove_top_pop_flag, remove_custom_items_flag, return_scores)

class UserKNNCFRecommenderStacked(BaseUserSimilarityMatrixRecommender):
    """ UserKNN recommender"""

    RECOMMENDER_NAME = "UserKNNCFRecommenderStacked"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]


    def __init__(self, URM_train, URM_validation, verbose = True):
        self.URM_train_real = URM_train
        self.URM_validation = URM_validation
        self.users_validation = URM_validation.shape[0]
        self.users_train = URM_train.shape[0]
        super(UserKNNCFRecommenderStacked, self).__init__(sps.vstack([URM_train, URM_validation]), verbose = verbose)

    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting = "none", URM_bias = False, **similarity_args):

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(self.FEATURE_WEIGHTING_VALUES, feature_weighting))

        if URM_bias is not None:
            self.URM_train.data += URM_bias

        if feature_weighting == "BM25":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = okapi_BM_25(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        elif feature_weighting == "TF-IDF":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = TF_IDF(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        similarity = Compute_Similarity(self.URM_train.T, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)

        self.W_sparse = similarity.compute_similarity(start_col=self.users_train, end_col=self.users_train+self.users_validation)
        r = np.arange(self.users_train, self.users_train+self.users_validation)
        set_zeros_row_col(self.W_sparse,r,r)
        self.W_sparse = self.W_sparse.transpose()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')
        