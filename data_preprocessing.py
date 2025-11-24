"""
Load and preprocess Fish dataset.
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_preprocessor(categorical_features=["Species"], numeric_features=None):
    if numeric_features is None:
        numeric_features = ["Length1", "Length2", "Length3", "Height", "Width"]

    # numeric pipeline: impute missing with median then scale
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # categorical pipeline: impute missing then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    return preprocessor


def preprocess_df(df: pd.DataFrame, preprocessor=None):
    expected = {"Species", "Length1", "Length2", "Length3", "Height", "Width", "Weight"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"Dataframe must contain columns: {expected}")

    X = df[["Length1", "Length2", "Length3", "Height", "Width", "Species"]].copy()
    y = df["Weight"].copy()

    if preprocessor is None:
        preprocessor = build_preprocessor()
        X_trans = preprocessor.fit_transform(X)
    else:
        X_trans = preprocessor.transform(X)

    return X_trans, y, preprocessor
