from RecSys_Course_AT_PoliMi.Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from sklearn.preprocessing import normalize


class SimilarityMergerHybridRecommender(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "SimilarityMergerHybridRecommender"

    def __init__(self, URM_train, recommender_1, recommender_2):
        super(SimilarityMergerHybridRecommender, self).__init__(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        self.W_sparse = None

    def fit(self, weight_1=0.5, weight_2=0.5, normalize_1=True, normalize_2=True):
        if normalize_1:
            W_sparse_1 = normalize(self.recommender_1.W_sparse, axis=1, norm='l1')
        else:
            W_sparse_1 = self.recommender_1.W_sparse
        if normalize_2:
            W_sparse_2 = normalize(self.recommender_2.W_sparse, axis=1, norm='l1')
        else:
            W_sparse_2 = self.recommender_2.W_sparse
        self.W_sparse = weight_1 * W_sparse_1 + weight_2 * W_sparse_2
