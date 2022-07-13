#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13/03/19

@author: Simone Boglio
"""

from DressipiChallenge.Recommenders.Recommender_utils import check_matrix
from DressipiChallenge.Recommenders.BaseCBFRecommender import BaseUserCBFRecommender
from DressipiChallenge.Recommenders.BaseSimilarityMatrixRecommender import BaseUserSimilarityMatrixRecommender
from DressipiChallenge.Recommenders.IR_feature_weighting import okapi_BM_25, TF_IDF
from DressipiChallenge.Utils.set_zeros import set_zeros_row_col
import numpy as np
from scipy import sparse as sps
from DressipiChallenge.Recommenders.Similarity.Compute_Similarity import Compute_Similarity


class UserKNNCBFRecommender(BaseUserCBFRecommender, BaseUserSimilarityMatrixRecommender):
    """ UserKNN recommender"""

    RECOMMENDER_NAME = "UserKNNCBFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, URM_train, UCM_train, verbose = True):
        super(UserKNNCBFRecommender, self).__init__(URM_train, UCM_train, verbose = verbose)


    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting = "none", **similarity_args):

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(self.FEATURE_WEIGHTING_VALUES, feature_weighting))


        if feature_weighting == "BM25":
            self.UCM_train = self.UCM_train.astype(np.float32)
            self.UCM_train = okapi_BM_25(self.UCM_train)

        elif feature_weighting == "TF-IDF":
            self.UCM_train = self.UCM_train.astype(np.float32)
            self.UCM_train = TF_IDF(self.UCM_train)


        similarity = Compute_Similarity(self.UCM_train.T, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)

        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')

class UserKNNCBFRecommenderStacked(BaseUserCBFRecommender, BaseUserSimilarityMatrixRecommender):
    """ UserKNN recommender"""

    RECOMMENDER_NAME = "UserKNNCFRecommenderStacked"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]


    def __init__(self, URM_train, URM_validation, UCM_train, verbose = True):
        self.URM_train_real = URM_train
        self.URM_validation = URM_validation
        self.users_validation = URM_validation.shape[0]
        self.users_train = URM_train.shape[0]
        super(UserKNNCBFRecommenderStacked, self).__init__(sps.vstack([URM_train, URM_validation]), UCM_train, verbose = verbose)

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

        similarity = Compute_Similarity(self.UCM_train.T, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)

        self.W_sparse = similarity.compute_similarity(start_col=self.users_train, end_col=self.users_train+self.users_validation)
        r = np.arange(self.users_train, self.users_train+self.users_validation)
        set_zeros_row_col(self.W_sparse,r,r)
        self.W_sparse = self.W_sparse.transpose()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')
        

