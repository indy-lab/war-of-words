import numpy as np

from math import log, exp
from numba import jit


@jit(nopython=True)
def _exp(x):
    x = min(x, 500)
    return exp(x)


@jit(nopython=True)
def _logsumexp(xs):
    """Stable implementation of logarithm of a sum of exponentials."""
    xmax = max(xs)
    val = 0
    for x in xs:
        val += _exp(x - xmax)
    return log(val) + xmax


@jit(nopython=True)
def _compute_logits(X, params):
    """Compute logits given a feature matrix and some parameters."""
    return X.dot(params)


@jit(nopython=True)
def _compute_exp_logits(X, params):
    """Compute the numerators of a softmax, i.e., the exponentiated logits."""
    logits = _compute_logits(X, params)
    explogits = np.zeros_like(logits)
    for i, logit in enumerate(logits):
        explogits[i] = _exp(logit)
    return explogits


@jit(nopython=True)
def _compute_normalization(explogits):
    """Compute the normalization constant of a softmax given the explogits."""
    return explogits.sum()


@jit(nopython=True)
def probabilities_jit(X, params):
    explogits = _compute_exp_logits(X, params)
    Z = _compute_normalization(explogits)
    return explogits / Z


@jit(nopython=True)
def log_likelihood_obs(X, y, params):
    """Compute the log-likelihood for one observation (X, y), where X is the
    feature matrix of one observation and y is the class index.
    """
    logits = _compute_logits(X, params)
    logZ = _logsumexp(logits)
    return logits[y] - logZ


@jit(nopython=True)
def log_likelihood_jit(data, params):
    """Compute the log-likelihood of the parameters given the data."""
    llh = 0
    for X, y in data:
        llh += log_likelihood_obs(X, y, params)
    return llh


@jit(nopython=True)
def gradient_obs(X, y, params):
    """Compute the gradient for one observation (X, y), where X is the feature
    matrix of one observation and y is the class index.
    """
    probs = probabilities_jit(X, params)
    grad = np.zeros_like(params)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # For the feature corresponding to the index of the class.
            if i == y:
                grad[j] += X[i, j]
            # For all features, including the one corresponding to the class.
            grad[j] -= X[i, j] * probs[i]
    return grad


@jit(nopython=True)
def gradient_jit(data, params):
    """Compute the gradient of the log-likelihood."""
    grad = np.zeros_like(params)
    for X, y in data:
        grad += gradient_obs(X, y, params)
    return grad
