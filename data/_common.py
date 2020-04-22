import json
import numpy as np
import os
import pickle

from collections import Counter


def load_dataset(path):
    canonical = os.path.abspath(path)
    print(f'Loading canonical dataset {canonical}...')
    with open(canonical) as f:
        return [json.loads(l) for l in f.readlines()]


def _filter_dossiers(dataset, thr):
    # Count occurence of each dossiers.
    dossiers = list()
    for data in dataset:
        for datum in data:
            dossiers.append(datum['dossier_ref'])
    counter = Counter(dossiers)
    # Define list of dossiers to keep.
    keep = set([d for d, c in counter.items() if c > thr])
    k, d = len(keep), len(set(dossiers))
    print(f'Removed {d-k} ({(d-k)/d*100:.2f}%) dossiers.')
    return keep


def _filter_meps(dataset, thr):
    # Count occurence of each dossiers.
    meps = list()
    for data in dataset:
        for datum in data:
            for at in datum['authors']:
                meps.append(at['id'])
    counter = Counter(meps)
    # Define list of dossiers to keep.
    keep = set([d for d, c in counter.items() if c > thr])
    k, m = len(keep), len(set(meps))
    print(f'Removed {m-k} ({(m-k)/m*100:.2f}%) MEPs.')
    return keep


def filter_dataset(dataset, thr=10):
    """Remove dossiers with less than `thr` edits."""
    keep_doss = _filter_dossiers(dataset, thr)
    keep_mep = _filter_meps(dataset, thr)
    filtered_dataset = list()
    for data in dataset:
        kd, km = True, True
        for datum in data:
            if datum['dossier_ref'] not in keep_doss:
                kd = False
            if not all(at['id'] in keep_mep for at in datum['authors']):
                km = False
        if kd and km:
            filtered_dataset.append(data)
    d, f = len(dataset), len(filtered_dataset)
    print(f'Removed {d-f} ({(d-f)/d*100:.2f}%) conflicts.')
    print('Number of data points:', len(filtered_dataset))
    return filtered_dataset


def _shuffle(featmats, labels, seed):
    np.random.seed(seed)
    perm = np.random.permutation(range(len(featmats)))
    return [featmats[p] for p in perm], [labels[p] for p in perm]


def _split(array, split):
    thr = int(np.ceil(len(array)*split))
    return array[:thr], array[thr:]


def _save(feat, featmat, labels, path):
    data = {
        'features': feat,
        'feature_matrices': featmat,
        'labels': labels,
    }
    path = os.path.abspath(path)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved to {path}.')


def shuffle_split_save(features, featmats, labels, seed, split, output_path):
    if seed is not None:
        # Shuffle data.
        featmats, labels = _shuffle(featmats, labels, seed)

    if split is not None:
        # Split data.
        fmtrain, fmtest = _split(featmats, split)
        lbtrain, lbtest = _split(labels, split)
        # Save data.
        path = output_path.replace('.', '-train.')
        _save(features, fmtrain, lbtrain, path)
        path = output_path.replace('.', '-test.')
        _save(features, fmtest, lbtest, path)
    else:
        # Save data.
        _save(features, featmats, labels, output_path)
