import collections
import numpy as np


class Features:

    """Helper class to manage features."""

    def __init__(self):
        self._cnter = 0
        self._idx = dict()
        self._names = dict()
        self._groups = collections.defaultdict(list)

    def __len__(self):
        return self._cnter

    def __repr__(self):
        return self._idx.__repr__()

    def __str__(self):
        return self._idx.__str__()

    def add(self, feature_name, group='default'):
        if feature_name not in self._idx:
            self._idx[feature_name] = self._cnter
            self._names[self._cnter] = feature_name
            self._groups[group].append(self._cnter)
            self._cnter += 1

    def get_group(self, group):
        return np.array(self._groups[group], dtype=int)

    def get_idx(self, feature_name):
        return self._idx[feature_name]

    def get_name(self, idx):
        return self._names[idx]

    def new_vector(self):
        return FeatureVector(self)

    def new_parameters(self):
        return ParameterVector(self)


class Vector:

    """Base class for vectors."""

    def __init__(self, features: Features, base=None):
        if base is not None:
            self._vec = base.copy()
            if type(base) is np.ndarray:
                self._vec = self._vec.tolist()
        else:
            # self._vec = np.zeros(len(features), dtype=float)
            self._vec = [0] * len(features)
        self._features = features

    def __len__(self):
        return len(self._vec)

    def __setitem__(self, key, val):
        self._vec[self._features.get_idx(key)] = val

    def __getitem__(self, key):
        return self._vec[self._features.get_idx(key)]

    def get_group(self, group):
        return np.array(self._vec)[self._features.get_group(group)].tolist()

    def as_array(self):
        array = np.zeros(len(self._vec), dtype=float)
        for i, v in enumerate(self._vec):
            if v != 0:
                array[i] = v
        return array
        # return self._vec

    def as_sparse_list(self):
        return [(i, v) for i, v in enumerate(self._vec) if v != 0]


class FeatureVector(Vector):

    """Helper class to manage feature vectors."""


class ParameterVector(Vector):

    """Helper class to manage model parameters."""

    def l2norm(self):
        return np.sqrt(self._vec.dot(self._vec))
