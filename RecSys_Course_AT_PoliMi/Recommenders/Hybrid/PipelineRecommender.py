from RecSys_Course_AT_PoliMi.Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from RecSys_Course_AT_PoliMi.Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from RecSys_Course_AT_PoliMi.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from sklearn.preprocessing import normalize
import scipy.sparse as sps
import numpy as np

class PipelineRecommenderP3AlphaKNNCFCBF(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "PipelineRecommenderP3AlphaKNNCFCBF"

    def __init__(self, URM_train, ICM, verbose=True):
        self.ICM = ICM
        self.URM_elaborated = None
        self.rec1 = None
        self.rec2 = None
        self.weight = None
        super(PipelineRecommenderP3AlphaKNNCFCBF, self).__init__(URM_train, verbose=verbose)

    def fit(self, P3Alpha_topK, P3Alpha_alpha, P3Alpha_normalize_similarity,
            KNN_shrink, KNN_similarity, KNN_feature_weighting,
            KNN_topK, KNN_normalize, KNN_ICM_weight, weight):
        self.weight = weight
        self.rec1 = P3alphaRecommender(self.URM_train)
        self.rec1.fit(P3Alpha_topK, P3Alpha_alpha, P3Alpha_normalize_similarity)
        print('P3Alpha fitted')
        self.URM_elaborated = sps.csr_matrix(np.dot(self.URM_train, self.rec1.W_sparse))
        self.rec2 = ItemKNN_CFCBF_Hybrid_Recommender(self.URM_train, self.ICM)
        params = {'topK': KNN_topK, 'shrink': KNN_shrink,
                  'similarity': KNN_similarity, 'normalize': KNN_normalize,
                  'feature_weighting': KNN_feature_weighting}
        self.rec2.fit(KNN_ICM_weight, **params)
        print('ItemKNNCFCBF fitted')

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        rec1_scores = self.rec1._compute_item_score(user_id_array)
        rec2_scores = self.rec2._compute_item_score(user_id_array)
        scores = self.weight * rec1_scores + (1 - self.weight) * rec2_scores
        return np.array(scores)
