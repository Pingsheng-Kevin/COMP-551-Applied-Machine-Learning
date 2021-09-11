import numpy as np


class GaussianNB:

    def __init__(self, var_smoothing = 1e-9):
        self.var_smoothing = var_smoothing
        return

    def fit(self, X_train, y_train):
        # number of training point, dimension, class
        (N, D), C = X_train.shape, np.max(y_train) + 1
        # mean, variance
        theta_, sigma_ = np.zeros((C, D)), np.zeros((C, D))
        class_count_ = np.zeros(C, int)
        for c in range(C):
            x_c = X_train[y_train == c]
            class_count_[c] = x_c.shape[0]
            theta_[c, :] = np.mean(x_c, axis=0)
            sigma_[c, :] = np.var(x_c, axis=0)

        #My memory is limited; Can't directly do this
        #self.epsilon_ = self.var_smoothing * np.var(X_train, axis=0).max()
        vars = np.zeros(D)
        for i in range(D):
            vars[i] = np.var(X_train[:, i], axis=0)
        var_max = vars.max()
        self.epsilon_ = self.var_smoothing * var_max

        self.N = N
        self.D = D
        self.C = C
        self.class_prior_ = (class_count_ + 1) / (N + C)
        self.class_count_ = class_count_
        self.theta_ = theta_
        self.sigma_ = sigma_
        self.sigma_ += self.epsilon_
        self.X_train = X_train
        self.y_train = y_train
        return self
    def predict_log_proba(self, X):
        Nt = X.shape[0]
        #shape(C,)
        log_prior = np.log(self.class_prior_)
        #shape(Nt, C)
        log_posterior = np.zeros((Nt, self.C))
        for i in range(Nt):
            log_likelihood = -.5 * np.log(2 * np.pi * self.sigma_) - .5*(
                X[i, :] - self.theta_) ** 2 / self.sigma_
            sum_log_likelihood = np.sum(log_likelihood, axis=-1)
            # unormalized
            log_posterior[i, :] = log_prior + sum_log_likelihood
            # normalized
            log_posterior[i, :] = log_posterior[i, :] - GaussianNB._logsumexp(log_posterior[i, :])

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
    def _logsumexp(Z):
        Zmax = np.max(Z,axis=-1)                              # max over C
        log_sum_exp = Zmax + np.log(np.sum(np.exp(Z - Zmax), axis=-1))
        return log_sum_exp

    @staticmethod
    def evaluate_acc(y, yh):
        return np.sum(y == yh) / y.shape[0]
