import numpy as np


def ACTIVATION(actv):
    if actv == 'identity':
        return _identity
    elif actv == 'logistic':
        return _logistic
    elif actv == 'tanh':
        return _tanh
    elif actv == 'relu':
        return _relu
    else:
        return _softmax


def D_ACTIVATION(actv):
    if actv == 'identity':
        return _identity_d
    elif actv == 'logistic':
        return _logistic_d
    elif actv == 'tanh':
        return _tanh_d
    elif actv == 'relu':
        return _relu_d
    else:
        return _softmax_d


def _identity(z):
    res = z
    return res


def _logistic(z):
    res = 1. / (1. + np.exp(-z))
    return res


def _tanh(z):
    res = np.tanh(z)
    return res


def _relu(z):
    if np.isscalar(z):
        res = np.max((z, 0))
    else:
        zeros = np.zeros(z.shape)
        res = np.max(np.stack((z, zeros), axis=-1), axis=-1)

    return res


def _softmax(z):
    z = np.exp(z - np.max(z, axis=0))
    res = z / np.sum(z, axis=0)
    return res


def _identity_d(z):
    res = np.ones_like(z, dtype=float)
    return res


def _logistic_d(z):
    res = _logistic(z) * (1. - _logistic(z))
    return res


def _tanh_d(z):
    res = 1. - _tanh(z) ** 2
    return res


def _relu_d(z):
    res = 1. * (z > 0)
    return res


def _softmax_d(z):
    res = _softmax(z) * (1. - _softmax(z))
    return res
