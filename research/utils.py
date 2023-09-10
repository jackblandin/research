import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils.validation import check_is_fitted


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


def sklearn_clf_pipeline(feature_types, clf_inst):
    """
    Utility method with injectable boilerplate code for constructing a sklearn
    classifier pipeline. The input `clf_inst` doesn't need to be a sklearn
    classifier, but does need to adhere to the
    sklearn.base.BaseEstimator/ClassifierMixin paradigm.

    Parameters
    ----------
    feature_types : dict<str, list>
        Specifies which type of feature each column is; used for feature
        engineering. Keys are feature types ('boolean', 'categoric',
        'continuous', 'meta'). Values are lists of the columns with that
        feature type.
    clf_inst : sklearn.base.BaseEstimator, ClassifierMixin
        Sklearn classifier instance. E.g. `RandomForestClassifier()`.

    Returns
    -------
    pipe : sklearn.pipeilne.Pipeline
        scikit-learn pipeline.
    """
    numeric_trf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ('kbins', KBinsDiscretizer(
                n_bins=3, encode='ordinal', strategy='uniform')),
        ]
    )
    categoric_trf = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(
                handle_unknown="ignore", min_frequency=.01)),
            ("selector", SelectPercentile(chi2, percentile=90)),
        ]
    )
    transformers = []

    if len(feature_types['continuous']) > 0:
        transformers.append(("num", numeric_trf, feature_types['continuous']))

    if len(feature_types['categoric']) > 0:
        transformers.append(("cat", categoric_trf, feature_types['categoric']))

    if len(feature_types['boolean']) > 0:
        transformers.append(('bool', categoric_trf, feature_types['boolean']))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        sparse_threshold=.000001,
    )
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf_inst),
        ],
    )
    return pipe
