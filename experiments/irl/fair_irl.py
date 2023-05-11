import logging
import pandas as pd
from research.utils import *


def generate_adult_dataset(
        filepath, n=10_000, z_col='is_race_white', y_col='is_income_over_50k',
):
    """
    Wrapper function for generating a sample of the adult dataset. This
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

    # Import dataset
    df = pd.read_csv(filepath)

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

    # Display useful summary info on z and y
    logging.info('Dataset count of each z, y group')
    logging.info(
        df_to_log(
            df.groupby(['z'])[['y']].agg(['count', 'mean'])
      )
    )

    # Split into inputs and target variables
    y = df['y']
    X = df.copy().drop(columns='y')

    return df, X, y



def generate_compas_dataset(
        filepath, n=10_000, z_col='is_race_white', y_col='is_recid',
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

    # Display useful summary info on z and y
    logging.info('Dataset count of each z, y group')
    logging.info(
        df_to_log(
            df.groupby(['z'])[['y']].agg(['count', 'mean'])
      )
    )

    # Split into inputs and target variables
    y = df['y']
    X = df.copy().drop(columns='y')

    return df, X, y
