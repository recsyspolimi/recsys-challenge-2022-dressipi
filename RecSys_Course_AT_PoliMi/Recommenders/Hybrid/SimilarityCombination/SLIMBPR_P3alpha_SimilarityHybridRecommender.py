import scipy.sparse as sps
import numpy as np

from RecSys_Course_AT_PoliMi.Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from RecSys_Course_AT_PoliMi.Recommenders.SLIM.SLIMBPRRecommender import SLIMBPRRecommender
from RecSys_Course_AT_PoliMi.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender


class SLIMBPR_P3alpha_SimilarityHybridRecommender(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "SLIMBPR_P3alpha_SimilarityHybridRecommender"

    def __init__(self, URM_train):
        super(SLIMBPR_P3alpha_SimilarityHybridRecommender, self).__init__(URM_train)
        self.URM_train = sps.csr_matrix(URM_train)
        self.slimbpr_recommender = SLIMBPRRecommender(URM_train)
        self.p3alpha_recommender = P3alphaRecommender(URM_train)

    def fit(self, alpha=0.5,
            slimbpr_topK=100, slimbpr_epochs=25, slimbpr_lambda_i=0.0025,
            slimbpr_lambda_j=0.00025, slimbpr_learning_rate=0.05,
            p3alpha_topK=100, p3alpha_alpha=1., p3alpha_normalize_similarity=False):

        self.slimbpr_recommender.fit(
            topK=slimbpr_topK,
            epochs=slimbpr_epochs,
            lambda_i=slimbpr_lambda_i,
            lambda_j=slimbpr_lambda_j,
            learning_rate=slimbpr_learning_rate)

        self.p3alpha_recommender.fit(
            topK=p3alpha_topK,
            alpha=p3alpha_alpha,
            min_rating=0,
            implicit=True,
            normalize_similarity=p3alpha_normalize_similarity)

        slimbpr_W_sparse = self.slimbpr_recommender.W_sparse
        p3alpha_W_sparse = self.p3alpha_recommender.W_sparse

        slimbpr_max = slimbpr_W_sparse.max()
        slimbpr_min = slimbpr_W_sparse.min()
        p3alpha_max = p3alpha_W_sparse.max()
        p3alpha_min = p3alpha_W_sparse.min()

        slimbpr_W_sparse = slimbpr_W_sparse / (slimbpr_max - slimbpr_min)
        p3alpha_W_sparse = p3alpha_W_sparse / (p3alpha_max - p3alpha_min)

        self.W_sparse = alpha * slimbpr_W_sparse + (1 - alpha) * p3alpha_W_sparse

    def set_URM_train(self, URM_train_new, **kwargs):
        self.slimbpr_recommender.URM_train = URM_train_new
        self.p3alpha_recommender.URM_train = URM_train_new
