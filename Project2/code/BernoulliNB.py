import numpy as np


class BernoulliNB:
    def __init__(self, alpha=.01, binarize=.0, fit_prior=True):
        self.binarize = binarize
        self.alpha = alpha
        self.fit_prior = fit_prior
        return

    def fit(self, X_train, y_train):
        (N, D), C = X_train.shape, np.max(y_train) + 1
        # binarize the dataset
        if (self.binarize is not None):
            X_train[X_train <= self.binarize] = 0
            X_train[X_train > self.binarize] = 1

            print(X_train)
        class_count_ = np.zeros(C, int)
        feature_count_ = np.zeros((C, D), int)

        for c in range(C):
            x_c = X_train[y_train == c]
            class_count_[c] = x_c.shape[0]
            feature_count_[c, :] = np.sum(x_c, axis=0)

        self.N = N
        self.D = D
        self.C = C
        self.class_prior_ = 1 / C if self.fit_prior else class_count_ / N
        self.class_count_ = class_count_
        self.feature_count_ = feature_count_
        self.X_train = X_train
        self.y_train = y_train

    def predict_log_proba(self, X):
        log_prior = np.log(self.class_prior_)
        # shape(C, D)
        feature_prob = (self.feature_count_ + self.alpha) / (self.class_count_[:, None] + self.N * self.alpha)
        # shape(Nt, C, D)
        log_likelihood = np.log(feature_prob[None, :, :] * X[:, None, :] + (1 - feature_prob[None, :, :]) * (1 - X[:, None, :]))
        # shape(Nt, C)
        sum_log_likelihood = np.sum(log_likelihood, axis=-1)
        # unormalized, shape(Nt, C)
        log_posterior = log_prior + sum_log_likelihood
        # normalize
        log_posterior = log_posterior - BernoulliNB._logsumexp(log_posterior)[:, None]

        return log_posterior

    def predict_proba(self, X):
        log_posterior = self.predict_log_proba(X)
        posterior = np.exp(log_posterior)
        return posterior

    def predict(self, X):
        X_prob = self.predict_proba(X)
        X_pred = self.y_train[np.argmax(X_prob, axis=-1)]
        return X_pred

    def score(self, X_test, y_test):
        return (self.predict(X_test) == y_test) / y_test.shape[0]

    @staticmethod
    def _logsumexp(Z):
        # dimension of Z : Nt * C
        Z_max = np.max(Z, axis=-1)
        log_sum_exp = Z_max + np.log(np.sum(np.exp(Z - Z_max[:, None]), axis=-1))

        # dimension of log_sum_exp : (Nt, )
        return log_sum_exp

    @staticmethod
    def evaluate_acc(y, yh):
        return np.sum(y == yh) / y.shape[0]