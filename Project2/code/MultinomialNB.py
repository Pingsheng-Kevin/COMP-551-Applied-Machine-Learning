import numpy as np


class MultinomialNB:
    def __init__(self, alpha=.01, fit_prior=True):
        self.alpha = alpha
        self.fit_prior = fit_prior
        return

    def fit(self, X_train, y_train):
        (N, D), C = X_train.shape, np.max(y_train) + 1
        # This counts all the words i.e. the sum of all features for each class; different from the Bernoulli
        class_count_ = np.zeros(C, int)
        feature_count_ = np.zeros((C, D), int)

        for c in range(C):
            x_c = X_train[y_train == c]
            class_count_[c] = x_c.shape[0]
            feature_count_[c, :] = np.sum(x_c, axis=0)

        self.N = N
        self.D = D
        self.C = C
        self.class_prior = 1 / C if self.fit_prior else (class_count_ + 1) / (N + C)
        self.feature_count_ = feature_count_
        self.class_count_ = class_count_
        self.X_train = X_train
        self.y_train = y_train

    def predict_log_proba(self, X):
        Nt = X.shape[0]
        log_posterior = np.zeros((Nt, self.C))
        log_prior = self.class_prior
        # shape (C,)
        total_words = np.zeros(self.C)
        for c in range(self.C):
            total_words[c] = np.sum(self.X_train[self.y_train == c])
        theta_ = (self.feature_count_ + self.alpha) / (total_words[:, None] + self.N * self.alpha)

        for i in range(Nt):
            #log_likelihood = np.log(theta_[None, :, :] ** X[:, None, :])
            log_likelihood = X[i, None, :] * np.log(theta_)
            # X_sum = np.sum(X, axis=-1)
            # log_fact_sum = MultinomialNB._log_fact(X_sum)
            # sum_log_fact = np.sum(MultinomialNB._log_fact(X), axis=-1)
            # sum_log_likelihood = log_fact_sum[:, None] - sum_log_fact[:, None] + np.sum(log_likelihood, axis=-1)
            sum_log_likelihood = np.sum(log_likelihood, axis=-1)

            log_posterior[i, :] = log_prior + sum_log_likelihood
            log_posterior[i, :] = log_posterior[i, :] - MultinomialNB._logsumexp(log_posterior[i, :])

        return log_posterior

    def predict_proba(self, X):
        log_posterior = self.predict_log_proba(X)
        posterior = np.exp(log_posterior)
        return posterior

    def predict(self, X):
        X_prob = self.predict_proba(X)
        X_pred = np.argmax(X_prob, axis=-1)
        return X_pred

    def score(self, X_test, y_test):
        return np.sum(self.predict(X_test) == y_test) / y_test.shape[0]

    @staticmethod
    def _log_fact(X):
        with np.nditer(X, op_flags=['readwrite']) as it:
            for x in it:
                # x[...] = np.math.factorial(x)
                if (x <= 1):
                    x[...] = 0
                else:
                    log_sum = 0
                    for i in range(2, x + 1):
                        log_sum += np.log(i)
                    x[...] = log_sum
        return X

    @staticmethod
    def _logsumexp(Z):
        # dimension of Z : Nt * C
        Z_max = np.max(Z, axis=-1)
        log_sum_exp = Z_max + np.log(np.sum(np.exp(Z - Z_max), axis=-1))

        # dimension of log_sum_exp : (Nt,)
        return log_sum_exp

    @staticmethod
    def evaluate_acc(y, yh):
        return np.sum(y == yh) / y.shape[0]
