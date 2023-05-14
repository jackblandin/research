import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class ManualClassifier():
    """
    A Scikit-Learn compatible classifier that allows for manual predictions.

    Parameters
    ----------
    pred_lambda : lambda
        Lambda that takes in an input `X` row and returns a prediction. E.g.
        ```
        lambda row: int(row['gender'] == 'Female')
        ```

    Attributes
    ----------
    is_fitted_ : bool
        Used so that downstream apps can determine if this is a fitted
        classifier.
    """

    def __init__(self, pred_lambda):
        self.pred_lambda = pred_lambda
        self.is_fitted_ = True

    def fit(self, X, y):
        """
        Pass through. Doesn't do anything.

        Parameters
        ----------
        X : pandas.DataFrame
            Classification input.
        y : pandas.Series<int>
            Binary target variable.

        Returns
        -------
        None
        """
        pass

    def predict(self, X):
        """
        Selects target variable.
            ```
        Parameters
        ---------
        X : pandas.DataFrame
            Input data.

        Returns
        -------
        predictions : numpy.array<int>, len(len(X))
            The "predictions".
        """
        X = pd.DataFrame(X)
        actions = X.apply(self.pred_lambda, axis=1)
        return actions

    def predict(self, X):
        """
        Selects target variable.
            ```
        Parameters
        ---------
        X : pandas.DataFrame
            Input data.

        Returns
        -------
        predictions : numpy.array<int>, len(len(X))
            The "predictions".
        """
        X = pd.DataFrame(X)
        actions = X.apply(self.pred_lambda, axis=1)
        return actions

def is_clf_fitted(clf_inst):
    """
    Checks if a sklearn-like classifier is already fitted.

    Parameters
    ----------
    clf_inst : <sklearn.base.ClassifierMixin>
        Classifier to check wheither it's fitted.

    Returns
    -------
    is_fitted : bool
        True if clf_inst fitted. False if not fitted..
    """
    is_fitted = [
        v for v in vars(clf_inst) if (
            v.endswith("_") and not v.startswith("__")
        )
    ]
    return is_fitted

def df_to_log(df, title='', tab_level=1):
    tab_prefix = tab_level * '\t'
    str_out = (
        title
        + '\n'
        + tab_prefix
        + df.to_string().replace('\n', '\n' + tab_prefix)
        + '\n'
    )
    return str_out

