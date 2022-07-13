from DressipiChallenge.Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
import numpy as np
import numpy.linalg as LA

class ScoreSegmentedHybrid3(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "ScoreSegmentedHybrid3"

    def __init__(self, URM_train, recommender_1, recommender_2, recommender_3):
        super(ScoreSegmentedHybrid3, self).__init__(URM_train)

        self.URM_train = URM_train.copy()
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        self.recommender_3 = recommender_3
        self.interactions = np.ediff1d(self.URM_train.indptr)

    def fit(self, weight_1_1, weight_2_1, weight_3_1, weight_1_2, weight_2_2, weight_3_2,
            weight_1_3, weight_2_3, weight_3_3, normalize_1, normalize_2, normalize_3,
            first_step=90, second_step=320):
        self.weight_1_1 = weight_1_1
        self.weight_2_1 = weight_2_1
        self.weight_3_1 = weight_3_1
        self.weight_sum_1 = weight_1_1 + weight_2_1 + weight_3_1
        self.weight_1_2 = weight_1_2
        self.weight_2_2 = weight_2_2
        self.weight_3_2 = weight_3_2
        self.weight_sum_2 = weight_1_2 + weight_2_2 + weight_3_2
        self.weight_1_3 = weight_1_3
        self.weight_2_3 = weight_2_3
        self.weight_3_3 = weight_3_3
        self.weight_sum_3 = weight_1_3 + weight_2_3 + weight_3_3

        self.normalize_1 = normalize_1
        self.normalize_2 = normalize_2
        self.normalize_3 = normalize_3

        self.first_step = first_step
        self.second_step = second_step

    def _compute_weighted_score(self, w1, w2, w3, num_int):
        if num_int < self.first_step:
            if self.normalize_1:
                w1 /= LA.norm(w1, 2)
                w2 /= LA.norm(w2, 2)
                w3 /= LA.norm(w3, 2)
            return (w1 * self.weight_1_1 + w2 * self.weight_2_1 + w3 * self.weight_3_1) / self.weight_sum_1
        elif num_int > self.second_step:
            if self.normalize_2:
                w1 /= LA.norm(w1, 2)
                w2 /= LA.norm(w2, 2)
                w3 /= LA.norm(w3, 2)
            return (w1 * self.weight_1_2 + w2 * self.weight_2_2 + w3 * self.weight_3_2) / self.weight_sum_2
        else:
            if self.normalize_3:
                w1 /= LA.norm(w1, 2)
                w2 /= LA.norm(w2, 2)
                w3 /= LA.norm(w3, 2)
            return (w1 * self.weight_1_3 + w2 * self.weight_2_3 + w3 * self.weight_3_3) / self.weight_sum_3

    def _compute_item_score(self, user_id_array, items_to_compute):
        item_weights = np.empty([len(user_id_array), self.URM_train.shape[1]])
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array, items_to_compute)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array, items_to_compute)
        item_weights_3 = self.recommender_3._compute_item_score(user_id_array, items_to_compute)

        for i in range(len(user_id_array)):
            item_weights[i] = self._compute_weighted_score(item_weights_1[i], item_weights_2[i], item_weights_3[i],
                                                           self.interactions[i])
        return item_weights

    def set_URM_train(self, URM_train_new, **kwargs):
        self.URM_train = URM_train_new
        self.recommender_1.URM_train = URM_train_new
        self.recommender_2.URM_train = URM_train_new
        self.recommender_3.URM_train = URM_train_new


class ScoreSegmentedHybrid2(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "ScoreSegmentedHybrid2"

    def __init__(self, URM_train, recommender_1, recommender_2):
        super(ScoreSegmentedHybrid2, self).__init__(URM_train)

        self.URM_train = URM_train.copy()
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        self.interactions = np.ediff1d(self.URM_train.indptr)

    def fit(self, weight_1_1, weight_2_1, weight_1_2, weight_2_2,
            weight_1_3, weight_2_3, normalize_1, normalize_2, normalize_3,
            first_step=90, second_step=320):
        self.weight_1_1 = weight_1_1
        self.weight_2_1 = weight_2_1
        self.weight_sum_1 = weight_1_1 + weight_2_1
        self.weight_1_2 = weight_1_2
        self.weight_2_2 = weight_2_2
        self.weight_sum_2 = weight_1_2 + weight_2_2
        self.weight_1_3 = weight_1_3
        self.weight_2_3 = weight_2_3
        self.weight_sum_3 = weight_1_3 + weight_2_3

        self.normalize_1 = normalize_1
        self.normalize_2 = normalize_2
        self.normalize_3 = normalize_3

        self.first_step = first_step
        self.second_step = second_step

    def _compute_weighted_score(self, w1, w2, num_int):
        if num_int < self.first_step:
            if self.normalize_1:
                w1 /= LA.norm(w1, 2)
                w2 /= LA.norm(w2, 2)
            return (w1 * self.weight_1_1 + w2 * self.weight_2_1) / self.weight_sum_1
        elif num_int > self.second_step:
            if self.normalize_2:
                w1 /= LA.norm(w1, 2)
                w2 /= LA.norm(w2, 2)
            return (w1 * self.weight_1_2 + w2 * self.weight_2_2) / self.weight_sum_2
        else:
            if self.normalize_3:
                w1 /= LA.norm(w1, 2)
                w2 /= LA.norm(w2, 2)
            return (w1 * self.weight_1_3 + w2 * self.weight_2_3) / self.weight_sum_3

    def _compute_item_score(self, user_id_array, items_to_compute):
        item_weights = np.empty([len(user_id_array), self.URM_train.shape[1]])
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array, items_to_compute)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array, items_to_compute)

        for i in range(len(user_id_array)):
            item_weights[i] = self._compute_weighted_score(item_weights_1[i], item_weights_2[i],
                                                           self.interactions[i])
        return item_weights

    def set_URM_train(self, URM_train_new, **kwargs):
        self.URM_train = URM_train_new
        self.recommender_1.URM_train = URM_train_new
        self.recommender_2.URM_train = URM_train_new
