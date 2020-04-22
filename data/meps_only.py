"""Process a dataset with authors only.

Format raw datapoints into feature vectors to be used with the WarOfWords
instances. You can optionally filter out dossiers and MEPs with less than a
given number of edits. You can also optionally shuffle the data and split them
into training and validation set. In this case, the output will be two files:
{output_file}-train.pkl and {output_file}-test.pkl. Otherwise, only
{output_file} is created.
"""

import argparse
import os

from _common import load_dataset, filter_dataset, shuffle_split_save
from warofwords import Features


def main(args):
    # Load dataset.
    path = os.path.abspath(args.canonical)
    dataset = load_dataset(path)

    # Filter dataset.
    if args.threshold is not None:
        print('Filtering dataset...')
        dataset = filter_dataset(dataset, thr=args.threshold)

    # Define features.
    features = Features()
    features.add('bias', group='bias')  # Add bias.
    for data in dataset:
        for datum in data:
            # Dossier feature.
            features.add(datum['dossier_ref'], group='dossier')
            # MEP features.
            for a in datum['authors']:
                features.add(a['id'], group='mep')

    print(f'There are {len(features)} features:')
    print(f'  - {len(features.get_group("mep"))} meps')
    print(f'  - {len(features.get_group("dossier"))} dossiers')

    # Build feature matrices and extract labels.
    print('Transforming dataset...')
    # Each data point is a feature matrix of N_k features, where N_k is the
    # total number of features (number of conflicting edits+1 for the dossier).
    featmats = list()
    labels = list()
    for data in dataset:
        featmat, label = list(), None
        # Extract labels and MEP features.
        for i, datum in enumerate(data):
            vec = features.new_vector()
            # Get MEP ids.
            for a in datum['authors']:
                vec[a['id']] = 1
            featmat.append(vec.as_sparse_list())
            # Add label if edit is accepted.
            if datum['accepted']:
                label = i

        # Add dossier features.
        vec = features.new_vector()
        vec[data[0]['dossier_ref']] = 1
        vec['bias'] = 1
        featmat.append(vec.as_sparse_list())

        # Add label if dossier wins conflict.
        if label is None:
            label = len(data)  # Set to the last one.
        labels.append(label)

        # Add feature matrix.
        featmats.append(featmat)

    shuffle_split_save(features, featmats, labels,
                       args.seed, args.split, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('canonical', help='Path to canonical dataset.')
    parser.add_argument('output_file', help='File to transformed dataset(s).')
    parser.add_argument('--threshold', default=None, type=int,
                        help='Filter dossiers with less than threshold edits.')
    parser.add_argument('--split', default=None, type=float,
                        help='Split into training and test sets (in [0, 1]).')
    parser.add_argument('--seed', default=None, type=int,
                        help='Seed for random generator.')
    args = parser.parse_args()

    main(args)
