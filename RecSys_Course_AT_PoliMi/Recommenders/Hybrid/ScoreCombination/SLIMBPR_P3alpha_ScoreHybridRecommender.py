import scipy.sparse as sps
import numpy as np
from DressipiChallenge.Recommenders.SLIM.SLIMBPRRecommender import SLIMBPRRecommender
from DressipiChallenge.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from DressipiChallenge.Recommenders.BaseRecommender import BaseRecommender


class SLIMBPR_P3alpha_ScoreHybridRecommender(BaseRecommender):
    RECOMMENDER_NAME = "SLIMBPR_P3alpha_ScoreHybridRecommender"

    def __init__(self, URM_train):
        super(SLIMBPR_P3alpha_ScoreHybridRecommender, self).__init__(URM_train)
        self.URM_train = sps.csr_matrix(URM_train)
        self.slimbpr_recommender = SLIMBPRRecommender(URM_train)
        self.p3alpha_recommender = P3alphaRecommender(URM_train)

    def fit(self, alpha=0.5,
            slimbpr_topK=100, slimbpr_epochs=25, slimbpr_lambda_i=0.0025,
            slimbpr_lambda_j=0.00025, slimbpr_learning_rate=0.05,
            p3alpha_topK=100, p3alpha_alpha=1., p3alpha_normalize_similarity=False):
        self.alpha = alpha

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

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        slimbpr_scores = self.slimbpr_recommender._compute_item_score(user_id_array)
        p3alpha_scores = self.p3alpha_recommender._compute_item_score(user_id_array)

        scores = self.alpha * slimbpr_scores + (1 - self.alpha) * p3alpha_scores

        scores = np.array(scores)

        return scores

    def set_URM_train(self, URM_train_new, **kwargs):
        self.slimbpr_recommender.URM_train = URM_train_new
        self.p3alpha_recommender.URM_train = URM_train_new
