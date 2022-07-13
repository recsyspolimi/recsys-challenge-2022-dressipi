import scipy.sparse as sps
import numpy as np
from RecSys_Course_AT_PoliMi.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from RecSys_Course_AT_PoliMi.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from RecSys_Course_AT_PoliMi.Recommenders.BaseRecommender import BaseRecommender


class ItemKNNCF_P3alpha_ScoreHybridRecommender(BaseRecommender):
    RECOMMENDER_NAME = "ItemKNNCF_P3alpha_ScoreHybridRecommender"

    def __init__(self, URM_train):
        super(ItemKNNCF_P3alpha_ScoreHybridRecommender, self).__init__(URM_train)
        self.URM_train = sps.csr_matrix(URM_train)
        self.itemknncf_recommender = ItemKNNCFRecommender(URM_train)
        self.p3alpha_recommender = P3alphaRecommender(URM_train)

    def fit(self, alpha=0.5,
            itemknncf_topK=50, itemknncf_shrink=100, itemknncf_similarity='cosine', itemknncf_normalize=True,
            itemknncf_feature_weighting="none", itemknncf_URM_bias=False,
            p3alpha_topK=100, p3alpha_alpha=1., p3alpha_normalize_similarity=False,
            **itemknncf_similarity_args):

        self.alpha = alpha

        self.itemknncf_recommender.fit(
            topK=itemknncf_topK,
            shrink=itemknncf_shrink,
            similarity=itemknncf_similarity,
            normalize=itemknncf_normalize,
            feature_weighting=itemknncf_feature_weighting,
            URM_bias=itemknncf_URM_bias,
            use_implementation="python",
            **itemknncf_similarity_args)

        self.p3alpha_recommender.fit(
            topK=p3alpha_topK,
            alpha=p3alpha_alpha,
            min_rating=0,
            implicit=True,
            normalize_similarity=p3alpha_normalize_similarity)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        itemknncf_scores = self.itemknncf_recommender._compute_item_score(user_id_array)
        p3alpha_scores = self.p3alpha_recommender._compute_item_score(user_id_array)

        scores = self.alpha * itemknncf_scores + (1 - self.alpha) * p3alpha_scores

        scores = np.array(scores)

        return scores

    def set_URM_train(self, URM_train_new, **kwargs):
        self.itemknncf_recommender.URM_train = URM_train_new
        self.p3alpha_recommender.URM_train = URM_train_new
