#! /bin/sh

# Train models for evaluation.
python train.py --data_path ../data/processed/meponly-ep7-train.pkl --model_path meponly-ep7.predict --regularizer 0.32
python train.py --data_path ../data/processed/rapadv-ep7-train.pkl --model_path rapadv-ep7.predict --regularizer 0.39
python train.py --data_path ../data/processed/meponly-ep8-train.pkl --model_path meponly-ep8.predict --regularizer 0.35
python train.py --data_path ../data/processed/rapadv-ep8-train.pkl --model_path rapadv-ep8.predict --regularizer 0.39

# Train models for parameter analysis.
python train.py --data_path ../data/processed/rapadv-ep7.pkl --model_path rapadv-ep7.fit --regularizer 0.39
python train.py --data_path ../data/processed/rapadv-ep8.pkl --model_path rapadv-ep8.fit --regularizer 0.39
