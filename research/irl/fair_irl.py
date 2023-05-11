import logging
import numpy as np
import pandas as pd
from research.rl.env.clf_mdp import *
from research.utils import *
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils.validation import check_is_fitted


def compute_demo_feature_exp(demo):
    """
    Transform demonstrations into feature expectations

    Parameters
    ----------
    demo : pandas.DataFrame
        Demonstrations. Each demonstration represents an iteration of a
        trained classifier and its predictions on a hold-out set. Columns are
            **`X` columns : all input columns (i.e. `X`)
            yhat : predictions
            y : ground truth targets

    Returns
    -------
    array<float>, len(2)
        mu0 : float, range(0,1)
            The accuracy feature expectations.
        mu1 : float, range(0,1)
            The fairness (disparate impact) feature expectations.
    """
    # Accuracy
    mu0 = np.mean(demo['yhat'] == demo['y'])

    # Disparate Impact
    p_yhat_eq_1_giv_z_eq_0 = ((demo['yhat'] == 1) & (demo['z'] == 0)).mean()
    p_yhat_eq_1_giv_z_eq_1 = ((demo['yhat'] == 1) & (demo['z'] == 1)).mean()

    mu1 = 1 - max([
        p_yhat_eq_1_giv_z_eq_0 - p_yhat_eq_1_giv_z_eq_1,
        p_yhat_eq_1_giv_z_eq_1 - p_yhat_eq_1_giv_z_eq_0,
    ])

    # Equal Opportunity
    p_yhat_eq_1_giv_y_eq_1_z_eq_0 = (
        ((demo['yhat'] == 1) & (demo['y'] == 1) & (demo['z'] == 0)).sum()
        / ((demo['y'] == 1) & (demo['z'] == 0)).sum()
    )
    p_yhat_eq_1_giv_y_eq_1_z_eq_1 = (
        ((demo['yhat'] == 1) & (demo['y'] == 1) & (demo['z'] == 1)).sum()
        / ((demo['y'] == 1) & (demo['z'] == 1)).sum()
    )
    print(p_yhat_eq_1_giv_y_eq_1_z_eq_0, p_yhat_eq_1_giv_y_eq_1_z_eq_1)
    mu2 = 1 - max([
        p_yhat_eq_1_giv_y_eq_1_z_eq_0 - p_yhat_eq_1_giv_y_eq_1_z_eq_1,
        p_yhat_eq_1_giv_y_eq_1_z_eq_1 - p_yhat_eq_1_giv_y_eq_1_z_eq_0,
    ])


    return np.array([
        mu0,
        mu1,
        mu2,
    ])


def compute_optimal_policy(
    clf_df, feature_types, clf_inst, x_cols, acc_weight, disp_imp_weight,
    # eq_opp_weight,
):
    """
    Learns the optimal policies from the provided reward weights.

    The `clf_df` is used to "fit" the ClfMDP parameters (e.g. b_eq, A_eq, etc)
    as well as the classifier that predicts `y` from `X`.

    The other classification parameters (`feature_types`, and `clf_inst` are
    used only on the classifier that predicts the `y` value from the `X`
    values.

    Parameters
    ----------
    clf_df : pandas.DataFrame
        Classification dataset. Required columns:
            'z' : int. Binary protected attribute.
            'y' : int. Binary target variable.
    feature_types : dict<str, list>
        Specifies which type of feature each column is; used for feature
        engineering. Keys are feature types ('boolean', 'categoric',
        'continuous', 'meta'). Values are lists of the columns with that
        feature type.
    clf_inst : sklearn.base.BaseEstimator, ClassifierMixin
        Sklearn classifier instance. E.g. `RandomForestClassifier()`. if
        unfitted, then `train_clf()` will be invoked with the provided
        `feature_types`. If fitted, then it is left as is.
    x_cols : list<str>
        The columns that are used in the state (along with `z` and `y`).
    acc_weight : float
        The Accuracy reward weight.
    disp_imp_weight : float
        The Disparate Impact reward weight.

    Returns
    -------
    opt_pol : list<np.array<int>>
        The optimal policy. If there are multiple, it randomly selects one.
    """
    gamma = 1e-6
    clf_mdp = ClassificationMDP(
        gamma=gamma,
        x_cols=x_cols,
        acc_reward_weight=acc_weight,
        disp_imp_reward_weight=disp_imp_weight,
        eq_opp_reward_weight=disp_imp_weight,
    )
    clf_mdp.fit(clf_df)
    optimal_policies = clf_mdp.compute_optimal_policies()
    sampled_policy = optimal_policies[np.random.choice(len(optimal_policies))]

    clf = train_clf(
        feature_types,
        clf_inst,
        clf_df.iloc[:, :-1],
        clf_df.iloc[:, -1],
    )
    clf_pol = ClassificationMDPPolicy(
        mdp=clf_mdp,
        pi=sampled_policy,
        clf=clf,
    )

    return clf_pol


def generate_demo(clf, X_test, y_test):
    """
    Create demonstration dataframe (columns are '**X', 'yhat', 'y') from a
    fitted classifer `clf`.

    Parameters
    ----------
    clf : fitted sklearn classifier
    X_test : pandas.DataFrame
    y_test : pandas.Series

    Returns
    -------
    demo : pandas.DataFrame
        Demonstrations. Each demonstration represents an iteration of a
        trained classifier and its predictions on a hold-out set. Columns are
            **`X` columns : all input columns (i.e. `X`)
            yhat : predictions
            y : ground truth targets
    """
    yhat = clf.predict(X_test)
    demo = pd.DataFrame(X_test).copy()
    demo['yhat'] = yhat
    demo['y'] = y_test.copy()
    return demo


def generate_expert_demonstrations(
        X_demo, y_demo, feature_types, clf_inst, n_feat_exp, m=3,
):
    """
    Generates the expert demonstrations which will be used as the positive
    training samples in the IRL loop. Each demonstration is a vector of
    feature expectations. Each demonstration fits a classifier on a fold of
    the provided dataset and then predicting on the holdout part of the fold.

    Parameters
    ----------
    X_demo : numpy.ndarray
        The X training data reserved for generating the demonstrations.
    y_demo : array-like
        The y training data reserved for generating the demonstrations.
    feature_types : dict<str, list>
        Specifies which type of feature each column is; used for feature
        engineering. Keys are feature types ('boolean', 'categoric',
        'continuous', 'meta'). Values are lists of the columns with that
        feature type.
    clf_inst : sklearn.base.BaseEstimator, ClassifierMixin
        Sklearn classifier instance. E.g. `RandomForestClassifier()`. if
        unfitted, then `train_clf()` will be invoked with the provided
        `feature_types`. If fitted, then it is left as is.
    n_feature_exp : float
        The number of feature expectations.
    m : int, default 3
        The number of demonstrations to generate.

    Returns
    -------
    muE : numpy.ndarray, shape(m, n_feature_exp)
        The expert demonstrations. Each demonstration is a vector of feature
        expectations.
    demos : list<pandas.DataFrame>
        A list of all the demonstration dataframes.
    """
    muE = np.zeros((m, n_feat_exp))  # expert demo feature expectations
    demos = []

    # Generate demonstrations (populate muE)
    def _generate_demo(muE, demos, k, X_train, X_test, y_train, y_test):
        logging.info(f"\tStaring iteration {k+1}/{m}")

        # Fit the classifier
        clf = train_clf(feature_types, clf_inst, X_train, y_train)

        logging.info('\t\tGenerating demo...')
        demo = generate_demo(clf, X_test, y_test)
        logging.info(
            df_to_log(
                demo.groupby(['z', 'y'])[['yhat']].agg(['count', 'mean'])
            )
        )

        logging.info('\t\tComputing feature expectations...')
        muE[k] = compute_demo_feature_exp(demo)
        demos.append(demo)
        logging.info(f"\t\tmuE[{k}]: {muE[k]}")

        return muE, demos

    logging.info('')
    logging.info('Generating expert demonstrations...')

    if m > 1:
        k_fold = KFold(m)
        for k, (train, test) in enumerate(k_fold.split(X_demo, y_demo)):
            X_train, y_train = X_demo.iloc[train], y_demo.iloc[train]
            X_test, y_test = X_demo.iloc[test], y_demo.iloc[test]
            muE, demos = _generate_demo(
                muE, demos, k, X_train, X_test, y_train, y_test,
            )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_demo, y_demo, test_size=.33,
        )
        muE, demos = _generate_demo(
            muE, demos, 0, X_train, X_test, y_train, y_test,
        )

    return muE, demos

def generate_initial_policies(
        X_demo, y_demo, feature_types, clf_inst, n_policies=1,
):
    """
    Generate initial policy(s) used as a starting point for the IRL loop SVM
    negative samples. These are the negative samples whereas
    `generate_expert_demonstrations()` produce the positive samples.

    Parameters
    ----------
    X_demo : pandas.DataFrame
        X data for fitting the initial policy.
    y_demo : pandas.Series
        y data for fitting the initial policy.
    feature_types : dict<str, list>
        Specifies which type of feature each column is; used for feature
        engineering. Keys are feature types ('boolean', 'categoric',
        'continuous', 'meta'). Values are lists of the columns with that
        feature type.
    clf_inst : sklearn.base.BaseEstimator, ClassifierMixin
        Sklearn classifier instance. E.g. `RandomForestClassifier()`. if
        unfitted, then `train_clf()` will be invoked with the provided
        `feature_types`. If fitted, then it is left as is.
    n_policies : int, default 1
        The number of policies to generate. Needs to be at least one.

    Returns
    -------
    mu : list<array<float>, len(2)> A list of:
        mu0 : float, range(0,1)
            The accuracy feature expectations.
        mu1 : float, range(0,1)
            The fairness (disparate impact) feature expectations.
    """
    mu = []

    logging.info('')
    for i in range(n_policies):
        logging.info(f"Generating initial policy {i+1}/{n_policies}")
        X_train, X_test, y_train, y_test = train_test_split(
            X_demo, y_demo, test_size=.33)

        logging.info('\tFitting classifier...')
        clf = train_clf(feature_types, clf_inst, X_train, y_train)

        logging.info('\tGenerating demo...')
        demo = generate_demo(clf, X_test, y_test)

        logging.info('\tComputing feature expectations...')
        feature_exp = compute_demo_feature_exp(demo)

        mu.append(feature_exp)
        logging.info(f"\tmu: {mu}")

    return mu


def irl_error(w, muE, muj):
    """
    Computes t[i] = wT(muE-mu[j])
    """
    mu_delta = muE.mean(axis=0) - muj
    l2_mu_delta = np.linalg.norm(mu_delta)
    l2_w = np.linalg.norm(w)
    err = l2_w * l2_mu_delta
    return err, mu_delta, l2_mu_delta


def train_clf(feature_types, clf_inst, X_train, y_train):
    """
    Trains a classifier, which corresponds to a demonstration (X, yhat, y).

    Parameters
    ----------
    feature_types : dict<str, list>
        Specifies which type of feature each column is; used for feature
        engineering. Keys are feature types ('boolean', 'categoric',
        'continuous', 'meta'). Values are lists of the columns with that
        feature type.
    clf_inst : sklearn.base.BaseEstimator, ClassifierMixin
        Sklearn classifier instance. E.g. `RandomForestClassifier()`. if
        unfitted, then `train_clf()` will be invoked with the provided
        `feature_types`. If fitted, then it is left as is.
    X_train : pandas.DataFrame
    y_train : pandas.Series

    Returns
    -------
    clf : sklearn classifier
        Fitted classifier.
    """
    # Check if classifier is already fitted
    logging.info('\t\tChecking if classifier already fitted...')
    fitted = [
        v for v in vars(clf_inst) if (
            v.endswith("_") and not v.startswith("__")
        )
    ]
    if fitted:
        logging.info('\t\tClassifier already fitted. Skipping fit.')
        return clf_inst

    logging.info('\t\tFitting classifier...')
    numeric_trf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ('kbins', KBinsDiscretizer(
                n_bins=3, encode='ordinal', strategy='uniform')),
        ]
    )
    categoric_trf = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ("selector", SelectPercentile(chi2, percentile=50)),
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
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf_inst),
        ],
    )
    clf.fit(X_train, y_train)
    return clf
