import logging
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from fairlearn.datasets import fetch_adult
from fairlearn.postprocessing import ThresholdOptimizer
from research.utils import *


def generate_dataset(dataset, n_samples):

    if dataset == 'Adult':
        X, y, feature_types = generate_adult_dataset(n=25_000)
    elif dataset == 'COMPAS':
        X, y, feature_types = generate_compas_dataset(n=25_000)

    X = X[
        feature_types['boolean']
        + feature_types['categoric']
        + feature_types['continuous']
    ]

    return X, y, feature_types


def generate_adult_dataset(
    n=10_000, z_col='is_race_white', y_col='is_income_over_50k',
):
    """
    Wrapper function for generating a sample of the adult dataset. This
    includes sampling down to just `n` samples, specifying the protected
    attribute 'z',

    Parameters
    ---------
    n : int, default 10_000
        Number of records to sample from dataset.
    z_col : str, default 'is_race_white'
        The column to use as the protected attribute. Must be binary.
    y_col : str, default 'is_income_over_50k'
        The column to use as the target variable. Must be binary.

    Returns
    -------
    df : pandas.DataFrame
        The X and y all as one dataframe.
    X : pandas.DataFrame
        The X (including z) columns.
    y : pandas.Series
        Just the y column.
    """
    data = fetch_adult(as_frame=True)
    df = data.data.copy()
    df['income'] = data.target.copy()

    # Take sample if possible
    if n < len(df):
        df = df.sample(n)

    # Common transformations
    df['is_income_over_50k'] = df['income'] == '>50K'
    df['is_race_white'] = df['race'] == 'White'

    # Specify the target variable `y`
    df['y'] = df[y_col].astype(int)

    # Specify the protected attribute `z`
    df['z'] = df[z_col].astype(int)

    # Display useful summary debug on z and y
    logging.debug('Dataset count of each z, y group')
    logging.debug(
        df_to_log(
            df.groupby(['z'])[['y']].agg(['count', 'mean'])
      )
    )

    # Split into inputs and target variables
    y = df['y']
    X = df.copy().drop(columns=['y', y_col, z_col, 'income'])

    # Balance the positive and negative classes
    rus = RandomUnderSampler(sampling_strategy=.5)
    X, y = rus.fit_resample(X, y)
    feature_types = {
        'boolean': [
            'z',
        ],
        'categoric': [
            'workclass',
            'education',
            # 'marital-status',
            # 'occupation',
            # 'relationship',
            # 'native-country',
            # 'race',
              'sex',
        ],
        'continuous': [
            # 'age',
            # 'educational-num',
            # 'capital-gain',
            # 'capital-loss',
            # 'hours-per-week',
        ],
        'meta': [
            'fnlwgt'
        ],
    }

    return X, y, feature_types


def generate_compas_dataset(
    n=10_000, z_col='is_race_white', y_col='is_recid',
    filepath='./../../data/compas/cox-violent-parsed.csv',
):
    """
    Wrapper function for generating a sample of the Compas dataset. This
    includes sampling down to just `n` samples, specifying the protected
    attribute 'z',

    Parameters
    ---------
    filepath : str
        Filepath for dataset.
    n : int, default 10_000
        Number of records to sample from dataset.
    z_col : str, default 'is_race_white'
        The column to use as the protected attribute. Must be binary.
    y_col : str, default 'is_recid'
        The column to use as the target variable. Must be binary.

    Returns
    -------
    df : pandas.DataFrame
        The X and y all as one dataframe.
    X : pandas.DataFrame
        The X (including z) columns.
    y : pandas.Series
        Just the y column.
    """

    # Import dataset
    df = pd.read_csv(filepath)

    # Take sample if possible
    if n < len(df):
        df = df.sample(n)

    # Filter out records where we don't know their compas risk score
    df = df.query('is_recid >= 0').copy()

    # Common transformations
    df['is_race_white'] = (df['race'] == 'Caucasian').astype(int)
    df = df.rename(columns={
        'sex': 'gender',
    })

    # Specify the target variable `y`
    df['y'] = df[y_col].astype(int)

    # Specify the protected attribute `z`
    df['z'] = df[z_col].astype(int)

    # Display useful summary debug on z and y
    logging.debug('Dataset count of each z, y group')
    logging.debug(
        df_to_log(
            df.groupby(['z'])[['y']].agg(['count', 'mean'])
      )
    )

    # Split into inputs and target variables
    y = df['y']
    X = df.copy().drop(columns='y')

    # Balance the positive and negative classes
    # rus = RandomUnderSampler(sampling_strategy=.5)
    # X, y = rus.fit_resample(X, y)

    feature_types = {
        'boolean': [
            'z',
        ],
        'categoric': [
            'gender',
            'age_cat',
            'score_text',
            'v_score_text',
        ],
        'continuous': [
            # 'age',
            # 'decile_score',
            # 'juv_fel_count',
            # 'priors_count',
            # 'v_decile_score',
        ],
        'meta': [
        ],
    }

    return X, y, feature_types


class ReductionWrapper():
    def __init__(self, clf, sensitive_features):
        self.clf = clf
        self.sensitive_features = sensitive_features

    def fit(self, X, y, sample_weight=None, **kwargs):
        self.clf.fit(
            X,
            y,
            sensitive_features=X[self.sensitive_features],
            **kwargs,
        )
        return self

    def predict(self, X, sample_weight=None, **kwargs):
        return self.clf.predict(
            X,
            sensitive_features=X[self.sensitive_features],
            **kwargs,
        )
