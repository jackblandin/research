import logging
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from fairlearn.datasets import fetch_adult, fetch_boston
from folktables import ACSDataSource, ACSIncome
from research.utils import *


def generate_dataset(dataset_name, n_samples):
    """
    Helper method that returns a dataset based on the specified label.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    n_samples : int
        Number of samples to return from the dataset.

    Returns
    -------
    X : pandas.DataFrame
        The X (including z) columns.
    y : pandas.Series
        Just the y column.
    feature_types : dict<str, array-like>
        Mapping of column names to their type of feature. Used to when
        constructing sklearn pipelines.
    """
    if dataset_name == 'Adult':
        X, y, feature_types = generate_adult_dataset(n_samples)
    elif dataset_name == 'COMPAS':
        X, y, feature_types = generate_compas_dataset(n_samples)
    elif dataset_name == 'Boston':
        X, y, feature_types = generate_boston_housing_dataset(n_samples)
    elif 'ACSIncome__' in dataset_name:
        state = dataset_name[-2:]
        X, y, feature_types = generate_acs_income(n_samples, state=state)
    else:
        raise ValueError(f"Unrecognized dataset name: {dataset_name}")

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
    X : pandas.DataFrame
        The X (including z) columns.
    y : pandas.Series
        Just the y column.
    feature_types : dict<str, array-like>
        Mapping of column names to their type of feature. Used to when
        constructing sklearn pipelines.
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

    # NOTE: 05/20/2023
    # Resampling messes up the feature expectations. Don't do this if you're
    # doing IRL.
    #
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
    X : pandas.DataFrame
        The X (including z) columns.
    y : pandas.Series
        Just the y column.
    feature_types : dict<str, array-like>
        Mapping of column names to their type of feature. Used to when
        constructing sklearn pipelines.
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

    # Balance the positive and negative classes
    # rus = RandomUnderSampler(sampling_strategy=.5)
    # X, y = rus.fit_resample(X, y)

    quantile_features = []
    for cont_feat in [
        # 'age',
        # 'decile_score',
        'juv_fel_count',
        'priors_count',
        # 'v_decile_score',
    ]:
        for q in [
                .1,
                .75,
                .9,
        ]:
            f = f"{cont_feat}__{q}"
            df[f] = (df[cont_feat] <= df[cont_feat].quantile(q))
            quantile_features.append(f)

    # Split into inputs and target variables
    y = df['y']
    X = df.copy().drop(columns='y')

    feature_types = {
        'boolean': [
            'z',
        ],# + quantile_features,
        'categoric': [
            'gender',
            'age_cat',
            'score_text',
            'v_score_text',
        ],
        'continuous': [
        ],
        'meta': [
        ],
    }

    return X, y, feature_types


def generate_boston_housing_dataset(n=10_000):
    """
    Wrapper function for generating a sample of the boston housing dataset.

    Parameters
    ---------
    n : int, default 10_000
        Number of records to sample from dataset.

    Returns
    -------
    X : pandas.DataFrame
        The X (including z) columns.
    y : pandas.Series
        Just the y column.
    feature_types : dict<str, array-like>
        Mapping of column names to their type of feature. Used to when
        constructing sklearn pipelines.
    """
    data = fetch_boston(as_frame=True)
    df = data.data.copy()
    df['LSTAT_binary'] = df['LSTAT'] >= df['LSTAT'].median()
    df['MEDV'] = data.target.copy()

    # Take sample if possible
    if n < len(df):
        df = df.sample(n)
    if n > len(df):
        df = df.sample(n, replace=True)

    # Specify the protected attribute `z`
    # Median value for Z
    df['z'] = (df['B'] >= 381.44).astype(int)

    quantile_features = []
    for cont_feat in [
            'B',
            'CRIM',
            'ZN',
            'RM',
            'LSTAT',
    ]:
        for q in [
                # .05,
                # .1,
                .25,
        ]:
            f = f"{cont_feat}__{q}"
            df[f] = (df[cont_feat] <= df[cont_feat].quantile(q))
            quantile_features.append(f)

    y = (df['MEDV'] >= df['MEDV'].quantile(.75)).astype(int).copy()
    X = df.drop(columns='MEDV')

    # Balance the positive and negative classes
    rus = RandomUnderSampler(sampling_strategy=.5)
    X, y = rus.fit_resample(X, y)
    feature_types = {
        'boolean': [
            'z',
        ] + quantile_features,
        'categoric': [
        ],
        'continuous': [
            # 'CRIM',  # per capita crime rate by town
            # 'ZN',  # prop of residential land zoned for lots over 25,000 sqft
            # 'INDUS',  # prop of non-retail business acres per town
            # 'CHAS',  # Charles River dummy var (= 1 if bounds river; else 0)
            # 'NOX',  # nitric oxides concentration (parts per 10 million)
            # 'RM',  # average number of rooms per dwelling
            # 'AGE',  # proportion of owner-occupied units built prior to 1940
            # 'DIS',  # weighted distances to five Boston employment centers
            # 'RAD',  # index of accessibility to radial highways
            # 'TAX',  # full-value property-tax rate per $10,000
            # 'PTRATIO',  # pupil-teacher ratio by town
            # 'B', # 1000(Bk - 0.63)^2 where Bk is the proportion of Black ppl
            # 'LSTAT',  # % lower status of the population
        ],
        'meta': [
        ],
        'target': [
            'MEDV',  # Median value of owner-occupied homes in $1000's],
        ],
    }

    return X, y, feature_types


def generate_acs_income(n=10_000, state=None):
    """
    Wrapper function for generating a sample of the folktable ACSIncome
    dataset.

    See Appendix B of https://arxiv.org/pdf/2108.04884.pdf for full feature
    details.

    Parameters
    ---------
    n : int, default 10_000
        Number of records to sample from dataset.
    state : str
        The US state to use.

    Returns
    -------
    X : pandas.DataFrame
        The X (including z) columns.
    y : pandas.Series
        Just the y column.
    feature_types : dict<str, array-like>
        Mapping of column names to their type of feature. Used to when
        constructing sklearn pipelines.
    """
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    data = data_source.get_data(states=[state], download=True)
    X, y, _ = ACSIncome.df_to_pandas(data)
    df = X.copy()
    df['y'] = y
    df['y'] = df['y'].astype(int)

    # Take sample if possible
    if n < len(df):
        df = df.sample(n)
    if n > len(df):
        df = df.sample(n, replace=True)

    # Specify the protected attribute `z`
    df['z'] = (df['RAC1P'] == 1).astype(int)

    df = df.fillna(-1)

    quantile_features = []
    for cont_feat in [
            'AGEP',
            # 'SCHL',
            # 'WKHP',
    ]:
        for q in [
                .25,
                .5,
                .75,
        ]:
            f = f"{cont_feat}__{q}"
            df[f] = (df[cont_feat] <= df[cont_feat].quantile(q))
            quantile_features.append(f)

    y = df['y'].copy()
    X = df.drop(columns=['RAC1P', 'y'])

    # Features
    # --------
    # AGEP (Age) : 0-99 integers., nullable
    # COW (Class of worker): 1-9 integers, nullable
    # SCHL (Educational attainment) : 1-24 integers, nullable
    # MAR (Marital status) : 1-5 integers, not nullable
    # OCCP (Occupation) : categoric, 529 distinct values
    # POBP (Place of birth) : categoric, 219 distinct values
    # WKHP (Usual hours worked per week) : 1-99, nullable
    # SEX (Sex) : integers, 1 Male, 2 Female
    # RAC1P (Recoded detailed race code)
    #   - 1: White alone
    #   - 2: Black or African American alone
    #   - 3: American Indian alone
    #   - 4: Alaska Native alone
    #   - 5: American Indian and Alaska Native tribes specified
    #   - 6: Asian alone
    #   - 7: Native Hawaiian and Other Pacific Islander alone
    #   - 8: Some Other Race alone
    #   - 9: Two or More Races

    feature_types = {
        'boolean': [
            'z',
            'SEX',
        ] + quantile_features,
        'categoric': [
            'COW',
            'MAR',
            'OCCP',
            'POBP',
        ],
        'continuous': [
        ],
        'meta': [
        ],
        'target': [
        ],
    }

    return X, y, feature_types
