"""
This script load the models and predict the target variable.
This also logs evaluation metric on the dataset if labels are provided.
"""

import os
import time
import argparse
from typing import Optional
import logging
import logging.config

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import ClassifierMixin
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('')


def feature_engineering(df, assets_filepath, training=False):
    # 0 is the most frequent value in 'Column0'
    df.loc[df["Column0"].isna()] = 0

    # Impute values
    impute_values = {
        "Column9": "median",
        "Column14": 0.0013506071853317501,
        "Column5": -0.0074686502841777,
        "Column4": 0.0621205346920356,
        "Column3": 0.27213270929052197,
        "Column15": 0.0033900985562439,
        "Column6": -0.407939121815475,
        "Column8": -0.28394554507395087,
    }
    for col in impute_values.keys():
        df[f'{col}_MI_flag'] = 0
        df.loc[df[col].isna(), f'{col}_MI_flag'] = 1
        if impute_values[col] == 'median':
            df.loc[df[col].isna(), col] = df[col].median()
        else:
            df.loc[df[col].isna(), col] = impute_values[col]
    
    # Transform negative values in column 1
    neg_col1 = (df['Column1'] < 0)
    df.loc[neg_col1, 'Column1'] = df['Column1'].max() - df.loc[neg_col1, 'Column1']
    
    # df["Column2"] = boxcox((df["Column2"] - df["Column2"].min()) / (df["Column2"].max() - df["Column2"].min()) + 1)[0]

    # Transform column 3 and 4
    df["Column3_4"] = df['Column3'] - df['Column4'] # non-commutative
    # df = df.drop(['Column3', 'Column4'], axis=1)

    # Add function of column 6 and 8
    df["Column6_8"] = df['Column6'] * df['Column8']

    # Load assets is not training
    assets = {
        'feature_cols': [],
        'isoforests': {},
    }
    if not training:
        assets = joblib.load(assets_filepath)

    # Make outlier indicator
    for col in ["Column6", "Column7", "Column8"]:
        if training:
            isoforest = IsolationForest(random_state=42)
            isoforest.fit(df[[col]])
            df[f'{col}_isof'] = isoforest.predict(df[[col]])
            assets["isoforests"][col] = isoforest
        
        else:
            df[f'{col}_isof'] = assets["isoforests"][col].predict(df[[col]])
    
    # Select features
    if training:
        feature_cols = list(set(df.columns) - {'ID', 'target', 'Column9'})
        assets["feature_cols"] = feature_cols
    else:
        feature_cols = assets['feature_cols']
    df = df[feature_cols]

    # Save assets if training
    if training:
        os.makedirs(os.path.dirname(assets_filepath), exist_ok=True)
        joblib.dump(assets, assets_filepath)

    return df


def prepare_dataset(
    X: str,
    assets_filepath: str,
    Y: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Load dataset
    X = pd.read_csv(X)

    # Do preprocessing
    X = feature_engineering(X, assets_filepath, training=False)

    # Fetch target if Y is given
    if Y is not None:
        Y = pd.read_csv(Y)
        Y = Y['target']

    # Return
    return X, Y


def load_model(model_filepath: str) -> tuple[list[ClassifierMixin], ClassifierMixin]:
    saved_obj = joblib.load(model_filepath)
    return saved_obj['primary_estimators'], saved_obj['final_estimator']


def infer(
    primary_estimators: list[ClassifierMixin],
    final_estimator: ClassifierMixin,
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert 0. <= threshold <= 1.

    # First predict using primary estimators on unmask data only
    mask = (X["Column18"]==0) | (X["Column1"]==2495)
    X_dash = np.vstack(tuple(estimator['model'].predict_proba(X[~mask])[:, 1] for estimator in primary_estimators)).T
    
    # Predict using final estimator
    y_prob = np.zeros(X.shape[0])
    y_pred = np.zeros(X.shape[0])
    
    y_prob[~mask] = final_estimator.predict_proba(X_dash)[:, 1]
    y_pred[~mask] = (y_prob[~mask] > threshold).astype(np.int32)

    return y_prob, y_pred


def evaluate(
    y_prob: pd.DataFrame,
    y_pred: pd.DataFrame,
    Y: Optional[pd.DataFrame] = None,
) -> None:
    # ROC-AUC score
    print(f"\nROC-AUC score: {roc_auc_score(y_score=y_prob, y_true=Y)}")

    # Classification report
    print("\nClassification Report:\n")
    print(classification_report(y_pred=y_pred, y_true=Y))

    sns.heatmap(confusion_matrix(y_pred=y_pred, y_true=Y), annot=True, fmt="g")
    plt.title("Confusion matrix (Test)")
    plt.xlabel("Model")
    plt.ylabel("Ground Truth")
    plt.show()


if __name__ == '__main__':
    # Setup arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-filepath', default='models/stackEnsembleV1.joblib')
    parser.add_argument('--assets-filepath', default='models/isoforestsFS.joblib')
    parser.add_argument('--savepath', default='output/predictions.csv')
    parser.add_argument('--X', default='input/Test_20/X_Test_Data_Input.csv')
    parser.add_argument('--Y', required=False)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    ####################### PREPARE DATASET #######################
    logger.info('Preparing dataset for the inference..')
    _st_time = time.time()

    X, Y = prepare_dataset(
        X = args.X,
        Y = args.Y,
        assets_filepath=args.assets_filepath
    )
    logger.info(f'Dataset prepared in {time.time() - _st_time:.2f}s')


    ####################### LOAD MODEL #######################
    logger.info('Loading model for inference..')
    _st_time = time.time()

    primary_estimators, final_estimator = load_model(model_filepath=args.model_filepath)
    logger.info(f'Successfully loaded model in {time.time() - _st_time:.2f}s')

    ####################### PREDICT TARGET #######################
    logger.info('Predicting target..')
    _st_time = time.time()
    y_prob, y_pred = infer(
        primary_estimators=primary_estimators,
        final_estimator=final_estimator,
        X=X,
        threshold=args.threshold,
    )

    X['prediction'] = y_pred
    X['confidence'] = y_prob
    X.to_csv(args.savepath, index=False)

    ####################### EVALUATE PERFORMANCE #######################
    if Y is not None:
        logger.info('Evaluating performance..')
        _st_time = time.time()
        evaluate(
            y_prob=y_prob,
            y_pred=y_pred,
            Y=Y,
        )

