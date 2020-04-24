# War of Words: The Competitive Dynamics of Legislative Processes

Data and code for

> Kristof, V., Grossglauser, M., Thiran, P., [*War of Words: The Competitive Dynamics of Legislative Processes*](https://infoscience.epfl.ch/record/275473/), The Web Conference, April 20-24, 2020, Taipei, Taiwan

## Set up

From the root of the repo, install the requirements and local library:

```
pip install -r requirements.txt
pip install -e lib
```

## Data

Download the (raw) data from https://zenodo.org/record/3757714.

## Processing

### For Training

Before [training the models](models/train.py), you first need to process the data and format them for input to the [model](lib/warofwords/model.py). Run the following commands to filter out dossiers and MEPs with less than 10 edits, split the data into 70/30 for training/testing, and fix the random seed:

```shell
# Train the WoW model.
python mep_only.py \
  path/to/war-of-words-ep7.txt \
  output/to/meponly-ep7.pkl \
  --threshold 10 \
  --split 0.7 \
  --seed 0

# Train the WoW(R) model.
python rapporteur_advantage.py \
  path/to/war-of-words-ep7.txt \
  output/to/rapadv-ep7.pkl \
  --threshold 10 \
  --split 0.7 \
  --seed 0
```

Do the same for EP8 by replacing `ep7` by `ep8` in the commands above.

This should have generated eight pickle files:

- `output/to/meponly-ep7-train.pkl`
- `output/to/meponly-ep7-test.pkl`
- `output/to/rapadv-ep7-train.pkl`
- `output/to/rapadv-ep7-test.pkl`
- `output/to/meponly-ep8-train.pkl`
- `output/to/meponly-ep8-test.pkl`
- `output/to/rapadv-ep8-train.pkl`
- `output/to/rapadv-ep8-test.pkl`

You can now use these files to train (and evaluate) the models (see below).

### For Parameter Analysis

We analyze the parameters after fitting the model on a *whole* legislature period, i.e., using the whole dataset for one legislature. This means that you need to generate another processed dataset, **without splitting**. We will do it only for the rapporteur-advantage model as this is our best model, hence providing the best estimation of the parameters. Run the following command:

```shell
python rapporteur_advantage.py \
  path/to/war-of-words-ep7.txt \
  output/to/rapadv-ep7.pkl \
  --threshold 10 \
  --seed 0
```

And do the same for EP8 by replacing `ep7` by `ep8` above.

This should have generated two additional pickle files:

- `output/to/rapadv-ep7.pkl`
- `output/to/rapadv-ep8.pkl`

You can now use these files to [analyze the parameters](notebooks/parameter-analysis.ipynb).

## Training

### For Evaluation

To train the models (`WoW` and `WoW(R)`), run the `models/train.py` script. It takes three arguments as parameters:

- `--data_path`: Path to a processed data set (see above)
- `--model_path`: Path where the trained model will be saved
- `--regularizer`: L2-regularizer for the model log-likelihood

Sorry, I forgot to report the value of the best regularizers (found by 10-fold cross-validation) in the paper. Here they are:

| \lambda     | EP7  | EP8  |
|-------------|------|------|
| `WoW`       | 0.32 | 0.35 |
| `WoW(R)`    | 0.39 | 0.39 |

You can then run the following command to train a model:

```shell
python train.py \
    --data_path ../data/processed/meponly-ep7-train.pkl \
    --model_path meponly-ep7.predict \
    --regularizer 0.32
```

I provide a bash script (`models/train-all-models.sh`) to train all the models in one go (make sure the path of the data files match).

### For Parameter Analysis

As explained above, you nee to train the models on the whole dataset to run the parameter analysis. Run this command to train the model on the whole EP7 legislature. (As a notation, I use the extensions `.predict` and `.fit` to differentiate between the models trained for evaluation (when you need to make predictions) and the models trained for parameter analysis (when you care only about fitting the model on all the data that you have).)

```shell
python train.py \
    --data_path ../data/processed/meponly-ep7.pkl \
    --model_path meponly-ep7.fit \
    --regularizer 0.32
```

## Evaluation

Run the [evaluation.ipynb](notebooks/evaluation.ipynb) notebook to reproduce the results of Table 2.

## Parameter Analysis

Run the [parameter-analysis.ipynb](notebooks/parameter-analysis.ipynb) notebook to reproduce the results of Table 3.

## Edit graph

Run the [edit-graph.ipynb](notebooks/edit-graph.ipynb) notebook to reproduce the results of Figure 3.

## Requirements

This project requires Python 3.6.

## Citation

To cite this work, use:

```
@inproceedings{kristof2020war,
  author = {Kristof, Victor and Grossglauser, Matthias and Thiran, Patrick},
  title = {War of Words: The Competitive Dynamics of Legislative Processes},
  year = {2020},
  booktitle = {Proceedings of The Web Conference 2020},
  pages = {2803–2809},
  numpages = {7},
  location = {Taipei, Taiwan},
  series = {WWW '20}
}
```
