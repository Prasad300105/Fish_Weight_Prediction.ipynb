# Fish Weight Prediction

Simple regression project to predict fish weight from physical measurements.

## Structure

```
Fish_Weight_Prediction_ipynb
├── data/
│   └── README.md
├── notebooks/
│   └── Fish_Weight_Prediction.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── utils.py
├── app/
│   └── main.py
├── models/           # (created after training)
├── results/          # (created after evaluation)
├── requirements.txt
└── .gitignore
```

## Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place `Fish.csv` into the `data/` folder (see `data/README.md` for expected columns).

3. Train model:

```bash
python src/train_model.py --data_path data/Fish.csv --output models/fish_model.joblib
```

4. Evaluate:

```bash
python src/evaluate.py --pipeline models/fish_model.joblib --data data/Fish.csv --out results
```

5. Run API:

```bash
uvicorn app.main:app --reload --port 8000
```

API POST `/predict` accepts JSON with fields:
`Length1, Length2, Length3, Height, Width, Species` and returns `predicted_weight`.
