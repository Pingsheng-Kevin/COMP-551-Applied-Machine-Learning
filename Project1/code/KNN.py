import numpy as np
import Dist
from KDTree import KDTree


class KNN(object):

    def __init__(self, K=1, dist_fn=Dist.euclidean, algorithm='brute', voting='uniform'):
        self.dist_fn = dist_fn
        self.K = K
        self.alg = algorithm
        self.voting = voting
        self.kd_tree = None
        return

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.C = np.max(y_train) + 1
        if self.alg == 'kdtree':
            self._build_kdtree()
        return self

    def _build_kdtree(self):
        k = self.x_train.shape[-1]
        self.kdtree = KDTree(k, self.x_train, self.dist_fn)

    def _kdtree_isbuild(self):
        return self.kdtree is not None

    def _get_Kneighbors(self, xi):
        nearest = self.kdtree.nearest_neighbour_search(xi, self.K)
        return nearest

    def kneighbors(self, x_test, return_distance=False):
        num_test = x_test.shape[0]
        knns = np.zeros((num_test, self.K), int)
        knds = np.zeros((num_test, self.K), float)
        if self.alg == 'kdtree':
            for i in range(num_test):
                knns[i, :] = list(map(lambda tup: tup[0], self._get_Kneighbors(x_test[i])))
                knds[i, :] = list(map(lambda tup: tup[-1], self._get_Kneighbors(x_test[i])))
        elif self.alg == 'brute':
            distances = self.dist_fn(self.x_train[None, :, :], x_test[:, None, :])
            for i in range(num_test):
                knns[i, :] = np.argsort(distances[i], axis=-1)[:self.K]
                knds[i, :] = np.sort(distances[i])[:self.K]

        if not return_distance:
            return knns
        return knns, knds

    def _predict_proba(self, x_test):
        num_test = x_test.shape[0]
        y_prob = np.zeros((num_test, self.C))

        if self.voting == 'uniform':
            knns = self.kneighbors(x_test)
            for i in range(num_test):
                y_prob[i, :] = np.bincount(self.y_train[knns[i, :]], minlength=self.C)
            y_prob /= self.K

        elif self.voting == 'distance':
            knns, knds = self.kneighbors(x_test, return_distance=True)
            for i in range(num_test):
                # check if any distance is zero
                k = self._check_dist(knds[i])
                if k != -1:
                    y_prob[i, self.y_train[knns[i, k]]] = 1
                else:
                    weights = list(map(lambda dist: 1 / dist, knds[i]))
                    y_prob[i, :] = np.bincount(self.y_train[knns[i, :]], weights=weights, minlength=self.C)
                    y_prob[i, :] /= np.sum(weights)

        return y_prob

    def predict(self, x_test):
        num_test = x_test.shape[0]
        y_pred = np.zeros(num_test, int)
        y_prob = self._predict_proba(x_test)

        y_pred[:] = np.argmax(y_prob, axis=-1)

        return y_pred

    @staticmethod
    def _check_dist(knd):
        # returns the index of distance which is zero, if not returns -1
        for k in range(len(knd)):
            if knd[k] == 0:
                return k
        return -1

    @staticmethod
    def evaluate_acc(y_target, y_true):
        correct = y_target == y_true
        return np.sum(correct) / y_target.shape[0]
