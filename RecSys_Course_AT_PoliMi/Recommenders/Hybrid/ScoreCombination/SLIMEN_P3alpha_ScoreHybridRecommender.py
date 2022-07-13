import scipy.sparse as sps
import numpy as np
from DressipiChallenge.Recommenders.SLIM.SLIMElasticNetRecommender_2 import MultiThread_SLIMElasticNetRecommender
from DressipiChallenge.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from DressipiChallenge.Recommenders.BaseRecommender import BaseRecommender


class SLIMEN_P3alpha_ScoreHybridRecommender(BaseRecommender):
    RECOMMENDER_NAME = "SLIMEN_P3alpha_ScoreHybridRecommender"

    def __init__(self, URM_train):
        super(SLIMEN_P3alpha_ScoreHybridRecommender, self).__init__(URM_train)
        self.URM_train = sps.csr_matrix(URM_train)
        self.slimen_recommender = MultiThread_SLIMElasticNetRecommender(URM_train)
        self.p3alpha_recommender = P3alphaRecommender(URM_train)

    def fit(self, alpha=0.5,
            slimen_alpha=1.0, slimen_l1_ratio=0.1, slimen_positive_only=True, slimen_topK=100,
            p3alpha_topK=100, p3alpha_alpha=1., p3alpha_normalize_similarity=False):
        self.alpha = alpha

        self.slimen_recommender.fit(
            alpha=slimen_alpha,
            l1_ratio=slimen_l1_ratio,
            positive_only=slimen_positive_only,
            topK=slimen_topK)

        self.p3alpha_recommender.fit(
            topK=p3alpha_topK,
            alpha=p3alpha_alpha,
            min_rating=0,
            implicit=True,
            normalize_similarity=p3alpha_normalize_similarity)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        slimen_scores = self.slimen_recommender._compute_item_score(user_id_array)
        p3alpha_scores = self.p3alpha_recommender._compute_item_score(user_id_array)

        scores = self.alpha * slimen_scores + (1 - self.alpha) * p3alpha_scores

        scores = np.array(scores)

        return scores

    def set_URM_train(self, URM_train_new, **kwargs):
        self.slimen_recommender.URM_train = URM_train_new
        self.p3alpha_recommender.URM_train = URM_train_new
