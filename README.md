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

Download the (raw) data from https://zenodo.org/record/3757714#.Xp3QTi_M0Wo.

## Processing

Process the (raw) data for the model.

To reproduce the results in the paper (filter out dossiers and MEPs with less than 10 edits, split 70/30 for training/testing, fix the random seed), run:

```
python meps_only.py \
  path/to/war-of-words-ep7.txt \
  output/processed/meponly-ep7.pkl \
  --threshold 10 \
  --split 0.7 \
  --seed 0
python rapporteur_advantage.py \
  path/to/war-of-words-ep7.txt \
  output/processed/rapadv-ep7.pkl \
  --threshold 10 \
  --split 0.7 \
  --seed 0
```

This should have generated four pickle files:

- `output/processed/meponly-ep7-train.pkl`
- `output/processed/meponly-ep7-test.pkl`
- `output/processed/rapadv-ep7-train.pkl`
- `output/processed/rapadv-ep7-test.pkl`

You can use these files to train and evaluate the models (see
[evaluation.ipynb](notebooks/evaluation.ipynb)).

Do the same for EP8 by replacing `ep7` by `ep8` in the commands above.

## Evaluation

Run the [evaluation.ipynb](notebooks/evaluation.ipynb) notebook to
reproduce the results of Table 2.

Run the [parameters.ipynb](notebooks/parameters.ipynb) notebook to reproduce the
results of Table 3.

## Edit graph

Run the [edit-graph.ipynb](notebooks/edit-graph.ipynb) notebook to reproduce the
results of Figure 3.

## Requirements

This project requires Python 3.6.
