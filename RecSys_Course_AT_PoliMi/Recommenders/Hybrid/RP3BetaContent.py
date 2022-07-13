import numpy as np

from RecSys_Course_AT_PoliMi.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
import scipy.sparse as sps


class RP3BetaContent(RP3betaRecommender):
    RECOMMENDER_NAME = "RP3betaContent"

    def __init__(self, URM_train, ICM_train, verbose=True):
        self.ICM_train = sps.csr_matrix(ICM_train.T, dtype=np.float32)
        print('ICM transposed, shape: {}'.format(self.ICM_train.shape))
        self.weight = 0
        super(RP3BetaContent, self).__init__(URM_train, verbose=verbose)

    def fit(self, alpha=1., beta=0.6, min_rating=0, topK=100, implicit=False, normalize_similarity=True, weight=0):
        self.weight = weight
        self.ICM_train *= self.weight
        self.URM_train = sps.vstack([self.URM_train, self.ICM_train])
        super(RP3BetaContent, self).fit(alpha, beta, min_rating, topK, implicit, normalize_similarity)
