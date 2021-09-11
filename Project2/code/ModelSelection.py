import numpy as np

k = 5


def kfoldCV(model, train_valid_sets):
    sum_acc = .0
    for (train_set, valid_set) in train_valid_sets:
        (X_train, y_train) = train_set
        (X_valid, y_valid) = valid_set
        model_fit = model.fit(X_train, y_train)
        yh_valid = model_fit.predict(X_valid)
        cmat = confusion_matrix(y_valid, yh_valid)
        sum_acc += np.sum(np.diag(cmat)) / np.sum(cmat)

    ave_acc = sum_acc / k
    return ave_acc


def cross_validation_split(X_train, y_train):
    n = X_train.shape[0]
    inds = np.random.permutation(n)
    n_val = n // k
    train_valid_sets = []
    for f in range(k):
        valid_inds = inds[f * n_val:: (f + 1) * n_val]
        train_inds = inds[0:: f * n_val] + inds[(f + 1) * n_val:: n]
        train_set = (X_train[train_inds], y_train[train_inds])
        valid_set = (X_train[valid_inds], y_train[valid_inds])
        train_valid_sets.append((train_set, valid_set))

    return train_valid_sets


def confusion_matrix(y, yh):
    n_classes = np.max(y) + 1
    c_matrix = np.zeros((n_classes, n_classes))
    for c1 in range(n_classes):
        for c2 in range(n_classes):
            c_matrix[c1, c2] = np.sum((y == c1) * (yh == c2))

    return c_matrix
