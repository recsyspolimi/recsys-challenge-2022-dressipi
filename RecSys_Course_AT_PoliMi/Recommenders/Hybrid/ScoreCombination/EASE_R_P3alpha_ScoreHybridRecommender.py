import scipy.sparse as sps
import numpy as np
from DressipiChallenge.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from DressipiChallenge.Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from DressipiChallenge.Recommenders.BaseRecommender import BaseRecommender


class EASE_R_P3alpha_ScoreHybridRecommender(BaseRecommender):
    RECOMMENDER_NAME = "EASE_R_P3alpha_ScoreHybridRecommender"

    def __init__(self, URM_train):
        super(EASE_R_P3alpha_ScoreHybridRecommender, self).__init__(URM_train)
        self.URM_train = sps.csr_matrix(URM_train)
        self.ease_r_recommender = EASE_R_Recommender(URM_train)
        self.p3alpha_recommender = P3alphaRecommender(URM_train)

    def fit(self, alpha=0.5,
            ease_r_topK=None, ease_r_normalize_matrix=False, ease_r_l2_norm=1e3,
            p3alpha_topK=100, p3alpha_alpha=1., p3alpha_normalize_similarity=False):

        self.alpha = alpha

        self.ease_r_recommender.fit(
            topK=ease_r_topK,
            normalize_matrix=ease_r_normalize_matrix,
            l2_norm=ease_r_l2_norm)

        self.p3alpha_recommender.fit(
            topK=p3alpha_topK,
            alpha=p3alpha_alpha,
            normalize_similarity=p3alpha_normalize_similarity)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        ease_r_scores = self.ease_r_recommender._compute_item_score(user_id_array)
        p3alpha_scores = self.p3alpha_recommender._compute_item_score(user_id_array)

        scores = self.alpha * ease_r_scores + (1 - self.alpha) * p3alpha_scores

        scores = np.array(scores)

        return scores