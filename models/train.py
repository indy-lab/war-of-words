import argparse
import json
import numpy as np
import os

from warofwords import WarOfWords, TrainedWarOfWords


def train(data_path, model_path, regularizer):
    """Train a model on a given dataset.

    Save the trained model to `model_path`.
    """

    # Load data.
    data_path = os.path.abspath(data_path)
    print(f'Loading {data_path}')
    features, featmats, labels = WarOfWords.load_data(data_path)
    train = list(zip(featmats, labels))

    # Initialize the model.
    hyperparams = WarOfWords.Hyperparameters(regularizer=regularizer)
    model = WarOfWords(train, features, hyperparams, bias_key='bias')

    # Train the model.
    print(f'Training the model (regularizer={regularizer})...')
    params, cost = model.fit()
    llh = model.log_likelihood(params['params'].as_array())
    print(f'Log-likelihood: {llh:.2f}')

    # Initialize a trained model.
    trained = TrainedWarOfWords(features, hyperparams, **params)

    # Save the trained model.
    model_path = os.path.abspath(model_path)
    trained.save(model_path)
    print(f'Saved model to {model_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to data')
    parser.add_argument('--model_path', required=True,
                        help='Path to saved model')
    parser.add_argument('--regularizer', type=float, required=True,
                        help='Regularizer',)
    args = parser.parse_args()

    train(args.data_path, args.model_path, args.regularizer)
