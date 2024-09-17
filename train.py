"""
This script utilises training data to train and save models.
It is expected to take 2-3 minutes for the hackathon dataset.
"""

import os
import time
import argparse
import logging
import logging.config

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('')


def load_dataset(
    trainX: str,
    trainY: str,
    testX: str,
    testY: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Read csv files
    df_X_train = pd.read_csv(trainX)
    df_y_train = pd.read_csv(trainY)
    df_X_test = pd.read_csv(testX)
    df_y_test = pd.read_csv(testY)

    return df_X_train, df_y_train, df_X_test, df_y_test


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
        logger.info(f'Saved assets at {assets_filepath}')

    return df


def prepare_dataset(
    trainX: str,
    trainY: str,
    testX: str,
    testY: str,
    assets_filepath: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Load dataset
    df_X_train, df_y_train, df_X_test, df_y_test = load_dataset(
        trainX = trainX,
        trainY = trainY,
        testX = testX,
        testY = testY,
    )

    # Do preprocessing
    df_X_train = feature_engineering(df_X_train, assets_filepath, training=True)
    df_X_test = feature_engineering(df_X_test, assets_filepath, training=False)

    # Return
    return df_X_train, df_y_train['target'], df_X_test, df_y_test['target']


def create_model() -> tuple[list[ClassifierMixin], ClassifierMixin]:
    """
    Creates a stacked classifier.
    """
    primary_estimators = [
        {
            'name': 'rf1',
            'sampling': False,
            'model': RandomForestClassifier(**{
                "n_estimators": 200,
                "max_depth": 25,
                "min_samples_split": 2,
                "min_samples_leaf": 3,
                "random_state": 42,
            })
        },
        {
            'name': 'rf2',
            'sampling': True,
            'model': RandomForestClassifier(**{
                "n_estimators": 400,
                "max_depth": 25,
                "min_samples_split": 9,
                "min_samples_leaf": 2,
                "random_state": 42,
            })
        },
        {
            'name': 'xgb1',
            'sampling': False,
            'model': xgb.XGBClassifier(**{
                "lambda": 0.0008484600529099388,
                "alpha": 1.6986081704936904e-05,
                "subsample": 0.6267017147154599,
                "colsample_bytree": 0.8715925884027383,
                "max_depth": 9,
                "min_child_weight": 3,
                "eta": 1.4251753380520105e-05,
                "gamma": 0.0005735117777982203,
                "random_state": 42
            })
        },
        {
            'name': 'xgb2',
            'sampling': True,
            'model': xgb.XGBClassifier(**{
                "lambda": 0.04838118613692709,
                "alpha": 1.6182815843808293e-08,
                "subsample": 0.9503265148901,
                "colsample_bytree": 0.9848560146737794,
                "max_depth": 9,
                "min_child_weight": 3,
                "eta": 0.23740845928648874,
                "gamma": 7.53138988543604e-06,
                "random_state": 42
            })
        },
        {
            'name': 'xgb3',
            'sampling': True,
            'model': xgb.XGBClassifier(**{
                "lambda": 3.5252596719136547e-07,
                "alpha": 3.091704773040277e-06,
                "subsample": 0.728324722716756,
                "colsample_bytree": 0.8566104914950033,
                "max_depth": 9,
                "min_child_weight": 3,
                "eta": 0.2944357417345176,
                "gamma": 0.20623355492809325,
                "random_state": 42
            })
        },
        {
            'name': 'catb1',
            'sampling': False,
            'model': CatBoostClassifier(**{
                "iterations": 700,
                "learning_rate": 0.05,
                "depth": 8,
                "l2_leaf_reg": 3e-2,
                "random_strength": 4e-5,
                "bagging_temperature": 0.45,
                "silent": True,
                "allow_writing_files": False
            })
        },
        {
            'name': 'catb2',
            'sampling': True,
            'model': CatBoostClassifier(**{
                "iterations": 700,
                "learning_rate": 0.05,
                "depth": 8,
                "l2_leaf_reg": 3e-2,
                "random_strength": 4e-5,
                "bagging_temperature": 0.45,
                "silent": True,
                "allow_writing_files": False
            })
        },
        {
            'name': 'catb3',
            'sampling': True,
            'model': CatBoostClassifier(**{
                "iterations": 867,
                "learning_rate": 0.025852814454489524,
                "depth": 10,
                "l2_leaf_reg": 17.468236424235297,
                "random_strength": 0.02082667862813585,
                "bagging_temperature": 0.05805895132755018,
                "silent": True,
                "allow_writing_files": False
            })
        },
        {
            'name': 'lgbm1',
            'sampling': False,
            'model': LGBMClassifier(**{
                "learning_rate": 0.15872997780904738,
                "num_leaves": 2920,
                "max_depth": 11,
                "min_data_in_leaf": 700,
                "lambda_l1": 0,
                "lambda_l2": 25,
                "min_gain_to_split": 0.09320318588121967,
                "bagging_fraction": 0.9,
                "bagging_freq": 1,
                "feature_fraction": 0.8,
                "random_state": 42,
                "force_col_wise": True,
                "verbose": -1
            })
        },
        {
            'name': 'lgbm2',
            'sampling': True,
            'model': LGBMClassifier(**{
                "learning_rate": 0.15872997780904738,
                "num_leaves": 2920,
                "max_depth": 11,
                "min_data_in_leaf": 700,
                "lambda_l1": 0,
                "lambda_l2": 25,
                "min_gain_to_split": 0.09320318588121967,
                "bagging_fraction": 0.9,
                "bagging_freq": 1,
                "feature_fraction": 0.8,
                "random_state": 42,
                "force_col_wise": True,
                "verbose": -1
            })
        },
    ]

    final_estimator = LogisticRegressionCV(cv=5, Cs=1, random_state=0, scoring='average_precision')

    return primary_estimators, final_estimator


def train(
    primary_estimators: list[ClassifierMixin],
    final_estimator: ClassifierMixin,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> None:
    # create dataset mask
    train_mask = (X_train["Column18"]==0) | (X_train["Column1"]==2495)
    test_mask = (X_test["Column18"]==0) | (X_test["Column1"]==2495)

    # sample dataset
    nomask_pos_counts = (y_train[~train_mask]==1).sum()
    nomask_neg_counts = (y_train[~train_mask]==0).sum()

    # Primary estimators
    for estimator in primary_estimators:
        if estimator['sampling'] and nomask_neg_counts < nomask_pos_counts:
            # Sample minority rows
            rows_to_sample = nomask_pos_counts - nomask_neg_counts
            sampling_class = 0 if rows_to_sample > 0 else 1

            x1 = X_train[(train_mask) & (y_train==sampling_class)].sample(abs(rows_to_sample), random_state=None)
            y1 = pd.Series(np.zeros(x1.shape[0]).astype(np.int64), name='target')

            X_train_new = pd.concat([X_train[~train_mask], x1])
            y_train_new = pd.concat([y_train[~train_mask], y1])

            # Fit model
            estimator['model'].fit(X_train_new, y_train_new)

        else:
            estimator['model'].fit(X_train, y_train)
        
        # Score model
        y_probs = estimator['model'].predict_proba(X_test[~test_mask])[:, 1]
        score = average_precision_score(y_score=y_probs, y_true=y_test[~test_mask])
        
        logger.info(f'PR-AUC on masked test dataset --> {estimator['name']}: {score:.4f}')
    
    # Predict using primary estimators
    X_train_dash = np.vstack(tuple(estimator['model'].predict_proba(X_train)[:, 1] for estimator in primary_estimators)).T
    X_test_dash = np.vstack(tuple(estimator['model'].predict_proba(X_test)[:, 1] for estimator in primary_estimators)).T

    # Train a final model on previous predictions 
    final_estimator.fit(X_train_dash, y_train)

    logger.info(f"Final estimator CV Score : {final_estimator.score(X_train_dash[~train_mask], y_train[~train_mask]):.4f}")
    logger.info(f"Final estimator PR-AUC (train) : {average_precision_score(y_score=final_estimator.predict_proba(X_train_dash[~train_mask])[:, 1], y_true=y_train[~train_mask]):.4f}")
    logger.info(f"Final estimator PR-AUC (test) : {average_precision_score(y_score=final_estimator.predict_proba(X_test_dash[~test_mask])[:, 1], y_true=y_test[~test_mask]):.4f}")


def save_models(
    primary_estimators: list[ClassifierMixin],
    final_estimator: ClassifierMixin,
    output_filepath: str,
) -> None:
    """
    Serialize models in pickle-based format
    Hence, this would require same environment as the training environement for inferencing.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    joblib.dump(
        {
            'primary_estimators': primary_estimators,
            'final_estimator': final_estimator
        },
        filename=output_filepath
    )
    logger.info(f'Saved model at {output_filepath}')


if __name__ == '__main__':
    # Setup arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainX', default='input/Train_60/X_Train_Data_Input.csv')
    parser.add_argument('--trainY', default='input/Train_60/Y_Train_Data_Target.csv')
    parser.add_argument('--testX', default='input/Test_20/X_Test_Data_Input.csv')
    parser.add_argument('--testY', default='input/Test_20/Y_Test_Data_Target.csv')
    parser.add_argument('--model-filepath', default='models/stackEnsembleV1.joblib')
    parser.add_argument('--assets-filepath', default='models/isoforestsFS1.joblib')
    args = parser.parse_args()

    ####################### PREPARE DATASET #######################
    logger.info('Preparing dataset for model training..')
    _st_time = time.time()

    X_train, y_train, X_test, y_test = prepare_dataset(
        trainX = args.trainX,
        trainY = args.trainY,
        testX = args.testX,
        testY = args.testY,
        assets_filepath=args.assets_filepath
    )
    logger.info(f'Dataset prepared in {time.time() - _st_time:.2f}s')

    ####################### BUILD MODEL #######################
    logger.info('Build model..')
    _st_time = time.time()

    primary_estimators, final_estimator = create_model()
    logger.info(f'Model building complete in {time.time() - _st_time:.2f}s')

    ####################### TRAIN MODELS #######################
    logger.info('Start training..')
    _st_time = time.time()

    train(
        primary_estimators=primary_estimators,
        final_estimator=final_estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    logger.info(f'Training complete in {time.time() - _st_time:.2f}s')

    ####################### SAVE MODELS #######################
    logger.info('Exporting model files..')
    _st_time = time.time()

    save_models(
        primary_estimators=primary_estimators,
        final_estimator=final_estimator,
        output_filepath=args.model_filepath,
    )
    logger.info(f'Export complete in {time.time() - _st_time:.2f}s')
