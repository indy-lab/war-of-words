import numpy as np
import pickle

from dataclasses import dataclass
from .features import ParameterVector
from ._model_jit import log_likelihood_jit, gradient_jit, probabilities_jit
from numba.typed import List
from scipy.optimize import minimize

# Remove Numba warnings.
from numba.errors import NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


class Model:

    """Base clase to define a model."""

    def __init__(self):
        pass

    @staticmethod
    def load_data(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        feature_matrices = list()
        for matrix in data['feature_matrices']:
            mat = list()
            for feat in matrix:
                vec = np.zeros(len(data['features']))
                for idx, val in feat:
                    vec[idx] = val
                mat.append(vec)
            feature_matrices.append(np.array(mat))
        features = data['features']
        labels = data['labels']

        return features, feature_matrices, labels


class TrainedModel(Model):

    """Base clase to define a trained model."""

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


class WarOfWords(Model):

    @dataclass
    class Hyperparameters:
        regularizer: float = 0.

    """Class for managing the WarOfWords model."""

    def __init__(self, data, features, hyperparameters, bias_key=None):
        """Initialize the model.

        :data: List of tuples of feature matrix and label index.
        :features: Instance of features.Features.
        :hyperparameters: Hyperparameters of the model.
        :bias_key: Optional. Key of bias in feautres.
        """
        super().__init__()
        # Format data for Numba.
        self._data = List()
        self._data.extend(data)
        self._features = features
        self._hyperparameters = hyperparameters
        if bias_key is not None:
            self._bias_idx = features.get_idx(bias_key)
        else:
            self._bias_idx = None

    def _gradient(self, params: np.ndarray) -> np.ndarray:
        """Compute the gradient of the *negative* log-likelihood with
        regularization, i.e., the objective function to minimize.
        """
        if self._bias_idx is not None:
            # Theta is the vector of parameters without the bias.
            theta = params.copy()
            theta[self._bias_idx] = 0.
        else:
            theta = params
        return (2 * self._hyperparameters.regularizer * theta
                - gradient_jit(self._data, params))

    def _objective(self, params: np.ndarray) -> float:
        """Compute the negative log-likelihood of the parameters with
        regularizer.
        """
        if self._bias_idx is not None:
            # Theta is the vector of parameters without the bias.
            b = self._bias_idx
            theta = np.concatenate((params[:b], params[b+1:]))
        else:
            theta = params
        return (self._hyperparameters.regularizer * np.sum(theta**2)
                - self.log_likelihood(params))

    def log_likelihood(self, params: np.ndarray) -> float:
        """Compute the log-likelihood of the parameters given the data."""
        return log_likelihood_jit(self._data, params)

    def fit(self, maxiter=15000, tol=1e-5, disp=False):
        x0 = self._features.new_parameters().as_array()
        options = {'gtol': tol,
                   'maxiter': maxiter,
                   'disp': disp}
        res = minimize(fun=self._objective,
                       x0=x0,
                       method='L-BFGS-B',
                       jac=self._gradient,
                       options=options)
        self.converged = res.success
        params = ParameterVector(self._features, base=res.x)
        return {'params': params}, res.fun


class TrainedWarOfWords(TrainedModel):

    def __init__(self, features, hyperparams, params):
        super().__init__()
        self.features = features
        self.hyperparameters = hyperparams
        self.parameters = params
        # Numpy array of params for faster computations.
        self._params = params.as_array()

    def probabilities(self, X):
        return probabilities_jit(X, self._params)

    def accuracy(self, data):
        acc = 0
        for X, y in data:
            prob = self.probabilities(X)
            if np.argmax(prob) == y:
                acc += 1
        return acc / len(data)

    def log_loss(self, data):
        loss = 0
        for X, y in data:
            prob = self.probabilities(X)
            loss -= np.log(prob[y])
        return loss / len(data)
