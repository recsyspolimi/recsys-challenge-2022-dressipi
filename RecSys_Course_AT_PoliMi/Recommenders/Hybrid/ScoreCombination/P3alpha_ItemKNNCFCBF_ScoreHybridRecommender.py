import scipy.sparse as sps
import numpy as np
from RecSys_Course_AT_PoliMi.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from RecSys_Course_AT_PoliMi.Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from RecSys_Course_AT_PoliMi.Recommenders.BaseRecommender import BaseRecommender


class P3alpha_ItemKNNCFCBF_ScoreHybridRecommender(BaseRecommender):
    RECOMMENDER_NAME = "P3alpha_ItemKNNCFCBF_ScoreHybridRecommender"

    def __init__(self, URM_train, ICM_train):
        super(P3alpha_ItemKNNCFCBF_ScoreHybridRecommender, self).__init__(URM_train)
        self.URM_train = sps.csr_matrix(URM_train)
        self.p3alpha_recommender = P3alphaRecommender(URM_train)
        self.itemknncfcbf_recommender = ItemKNN_CFCBF_Hybrid_Recommender(URM_train, ICM_train)

    def fit(self, alpha=0.5,
            p3alpha_topK=100, p3alpha_alpha=1., p3alpha_normalize_similarity=False,
            itemknncfcbf_topK=50, itemknncfcbf_shrink=100, itemknncfcbf_similarity='cosine',
            itemknncfcbf_normalize=True, itemknncfcbf_feature_weighting="none",
            itemknncfcbf_ICM_bias=None, itemknncfcbf_ICM_weight=1.0):
        self.alpha = alpha

        self.p3alpha_recommender.fit(
            topK=p3alpha_topK,
            alpha=p3alpha_alpha,
            normalize_similarity=p3alpha_normalize_similarity)

        self.itemknncfcbf_recommender.fit(
            topK=itemknncfcbf_topK,
            shrink=itemknncfcbf_shrink,
            similarity=itemknncfcbf_similarity,
            normalize=itemknncfcbf_normalize,
            feature_weighting=itemknncfcbf_feature_weighting,
            ICM_bias=itemknncfcbf_ICM_bias,
            ICM_weight=itemknncfcbf_ICM_weight,
            use_implementation="python"
        )

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        p3alpha_scores = self.p3alpha_recommender._compute_item_score(user_id_array)
        itemknncfcbf_scores = self.itemknncfcbf_recommender._compute_item_score(user_id_array)

        scores = self.alpha * p3alpha_scores + (1 - self.alpha) * itemknncfcbf_scores

        scores = np.array(scores)

        return scores

    def set_URM_train(self, URM_train_new, **kwargs):
        self.p3alpha_recommender.URM_train = URM_train_new
        self.itemknncfcbf_recommender.URM_train = URM_train_new
