import logging
import numpy as np
import os
import pandas as pd
from research.rl.env.clf_mdp import *
from research.rl.env.clf_mdp_policy import *
from research.rl.env.objectives import *
from research.utils import *
from sklearn.model_selection import train_test_split, KFold


def compute_optimal_policy(
    clf_df, clf, x_cols, obj_set, reward_weights, skip_error_terms=False,
    method='highs', gamma = 1e-9, min_freq_fill_pct=0, restrict_y=True,
):
    """
    Learns the optimal policies from the provided reward weights.

    The `clf_df` is used to "fit" the ClfMDP parameters (e.g. b_eq, A_eq, etc)
    as well as the classifier that predicts `y` from `X`.

    The other classification parameters (`feature_types`, and `clf` are
    used only on the classifier that predicts the `y` value from the `X`
    values.

    Parameters
    ----------
    clf_df : pandas.DataFrame
        Classification dataset. Required columns:
            'z' : int. Binary protected attribute.
            'y' : int. Binary target variable.
    clf : sklearn.pipeline.Pipeline
        Sklearn classifier instance.
    x_cols : list<str>
        The columns that are used in the state (along with `z` and `y`).
    obj_set : ObjectiveSet
        The objective set.
    reward_weights : dict<str, float>
        Keys are objective identifiers. Values are their respective reward
        weights.
    skip_error_terms : bool, default False
        If true, doesn't try and find all solutions and instead just invokes
        the scipy solver on the input terms.
    method : str, default 'highs'
        The scipy solver method to use. Options are 'highs' (default),
        'highs-ds', 'highs-ipm'.
    gamma : float, range, [0, 1)
        The MDP discount factor. Should be close to zero since this is a
        classification mdp, but cannot be zero exactly otherwise the linear
        program won't converge.
    min_freq_fill_pct, float, range[0, 1), default 0
        Minimum frequency for each input variable to not get replaced by a
        default value.
    restrict_y : bool, deafult True
        If True, policy must have same action for any x,z combo, regardless
        of y.

    Returns
    -------
    opt_pol : research.rl.env.clf_mdp.ClassificationMDPPolicy
        The optimal policy. If there are multiple, it randomly selects one.
    """
    clf_mdp = ClassificationMDP(
        gamma=gamma,
        obj_set=obj_set,
        x_cols=x_cols,
    )

    # Construct the mdp, including optimization problems
    clf_mdp.fit(
        reward_weights=reward_weights,
        clf_df=clf_df,
        min_freq_fill_pct=min_freq_fill_pct,
        restrict_y=restrict_y,
    )

    # Compute the optimal policy(s). This does NOT fit the classifier that
    # predicts Y from Z, X. That occurs on the generate_demo() call. This
    # `compute_optimal_policies` computes the optimal policy, assuming the
    # state is known.
    optimal_policies = clf_mdp.compute_optimal_policies(
        skip_error_terms=skip_error_terms,
        method=method,
    )

    # Pick from one of the policies (if there are multiple).
    logging.info(f"\n\t\tFound {len(optimal_policies)} optimal policies.")
    sampled_policy = optimal_policies[np.random.choice(len(optimal_policies))]

    clf_pol = ClassificationMDPPolicy(
        mdp=clf_mdp,
        pi=sampled_policy,
        clf=clf,
    )

    return clf_pol


def generate_demo(clf, X_test, y_test, can_observe_y=False):
    """
    Create demonstration dataframe (columns are '**X', 'yhat', 'y') from a
    fitted classifer `clf`.

    Parameters
    ----------
    clf : fitted sklearn classifier
    X_test : pandas.DataFrame
    y_test : pandas.Series
    can_observe_y : bool, default False
        Whether the policy can "see" y or if it needs to predict it from X.

    Returns
    -------
    demo : pandas.DataFrame
        Demonstrations. Each demonstration represents an iteration of a
        trained classifier and its predictions on a hold-out set. Columns are
            **`X` columns : all input columns (i.e. `X`)
            yhat : predictions
            y : ground truth targets
    """
    demo = pd.DataFrame(X_test).copy()

    if can_observe_y:
        # yhat = clf.predict(X_test, y_test)
        if 'y' in X_test.columns:
            demo['yhat'] = demo['y'].copy()
        else:
            demo['yhat'] = y_test
    else:
        demo['yhat'] = clf.predict(X_test)

    demo['y'] = y_test.copy()

    return demo


def generate_demos_k_folds( X, y, clf, obj_set, n_demos=3):
    """
    Generates the expert demonstrations which will be used as the positive
    training samples in the IRL loop, or generates the initial policy
    demonstrations that serve as the initial positive examples in the IRL loop.
    Each demonstration is a vector of feature expectations. Each demonstration
    fits a classifier on a fold of the provided dataset and then predicting on
    the holdout part of the fold.

    Parameters
    ----------
    X : numpy.ndarray
        The X training data reserved for generating the demonstrations.
    y : array-like
        The y training data reserved for generating the demonstrations.
    clf : sklearn.pipeline.Pipeline, ClassifierMixin
        Sklearn classifier pipeline.
    obj_set : ObjectiveSet
        The objective set.
    n_demos : int, default 3
        The number of demonstrations to generate (also the k in k folds).

    Returns
    -------
    mu : numpy.ndarray, shape(m, len(obj_set.objectives))
        The demonstrations feature expectations.
    demos : list<pandas.DataFrame>
        A list of all the demonstration dataframes.
    """
    mu = np.zeros((n_demos, len(obj_set.objectives)))  # demo feat exp
    demos = []

    # Generate demonstrations (populate mu)
    def _generate_demo(mu, demos, k, X_train, X_test, y_train, y_test):
        logging.debug(f"\tStaring iteration {k+1}/{n_demos}")

        # Fit the classifier
        clf.fit(X_train, y_train)

        logging.debug('\t\tGenerating demo...')
        demo = generate_demo(clf, X_test, y_test)
        logging.debug(
            df_to_log(
                demo.groupby(['z', 'y'])[['yhat']].agg(['count', 'mean'])
            )
        )

        logging.debug('\t\tComputing feature expectations...')
        mu[k] = obj_set.compute_demo_feature_exp(demo)
        demos.append(demo)
        logging.debug(f"\t\tmu[{k}]: {mu[k]}")

        return mu, demos

    logging.debug('')
    logging.debug('Generating expert demonstrations...')

    if n_demos > 1:
        k_fold = KFold(n_demos)
        for k, (train, test) in enumerate(k_fold.split(X, y)):
            _X_train, _y_train = X.iloc[train], y.iloc[train]
            _X_test, _y_test = X.iloc[test], y.iloc[test]
            mu, demos = _generate_demo(
                mu, demos, k, _X_train, _X_test, _y_train, _y_test,
            )
    else:
        _X_train, _X_test, _y_train, _y_test = train_test_split(
            X, y, test_size=.33,
        )
        mu, demos = _generate_demo(
            mu, demos, 0, _X_train, _X_test, _y_train, _y_test,
        )

    return mu, demos


def irl_error(
        w, muE, muL, dot_weights_feat_exp=True, allow_neg_weights=False,
):
    """
    Computes t[i] = argmax_{mu[j] for j in muL} wT(muE-mu[j])

    Parameters
    ----------
    w : np.array<float>
        Weights.
    muE : array-like<np.array<float>>, shape (n_expert_demos, len(len(objective_set.objectives)))
        Array of all expert feature expectations.
    muL : array-like<array-like<float>>
        List of all learned feature expectations.
    dot_weights_feat_exp : bool, default True
        If True, error = l2_norm( (muE - muL).dot(w) )
        If False, error = l2_norm(muE - muL) * l2_norm(w)
        The default value is the typical IRL one. Use the other if all feat
        exp should be non-zero weights. Helps avoid issue of IRL nulling out
        feat exp componets that it has trouble matching.
    allow_neg_weights : bool, default False
        If True, allows positive feature expectation errors to be nulled out
        even if weights are negative.

    Returns
    -------
    best_err : float
        The error of the best policy, and therefore the IRL error.
    best_j : int
        The index of the best policy in muL.
    mu_deltas[best_j] : array<float>
        The array of differences between muE and muL[best_j].
    l2_mu_delta[best_j] : float
        The l2 norm of the muE and muL deltas.
    """
    # mu_deltas = np.zeros((len(muL), muE.shape[1]))
    # l2_mu_deltas = np.zeros(len(muL))
    # best_err = np.inf
    # best_j = None

    # # Find best muj
    # for j, muj in enumerate(muL):
    #     # JDB 01/07/2024
    #     # Trying this out. Make mu delta errors RELATIVE to their magnitude. So
    #     # adding the muE.mean(axis=0) as a denominator
    #     mu_deltas[j] = muE.mean(axis=0) - muj
    #     # JDB 12/05/2023
    #     # Trying this out. If muj is better, reduce the amount it's considered
    #     # as an error for feature expectations deltas.
    #     # But first, check if all weights are positive. Don't want anything
    #     # to converge to zero error if weights are negative.
    #     if allow_neg_weights or  np.all(w > -1e-5):
    #         mu_deltas[j][mu_deltas[j] < 0] = 1 * mu_deltas[j][mu_deltas[j] < 0]

    #     if dot_weights_feat_exp:
    #         err = np.linalg.norm(np.abs(w) * mu_deltas[j], ord=2)
    #     else:
    #         err = np.linalg.norm(np.abs(mu_deltas[j])) * np.linalg.norm(w, ord=2)

    #     if err < best_err:
    #         best_err = err
    #         best_j = j

    # 01/08/2024
    # Trying this out: don't find best muj. Just compute error with most recent
    # muj and weights.
    l2_mu_deltas = np.zeros([])
    best_err = np.inf
    best_j = len(muL) - 1


    mu_deltas = (muE.mean(axis=0) - muL[-1]) / muE.mean(axis=0)

    # JDB 01/07/2024
    # Trying this out. Make mu delta errors RELATIVE to their magnitude. So
    # adding the muE.mean(axis=0) as a denominator
    if allow_neg_weights or  np.all(w > -1e-5):
        mu_deltas[mu_deltas < 0] = 1 * mu_deltas[mu_deltas < 0]

    if dot_weights_feat_exp:
        err = np.linalg.norm(np.abs(w) * mu_deltas, ord=1)
    else:
        err = np.linalg.norm(np.abs(mu_deltas)) * np.linalg.norm(w, ord=1)

    best_err = err

    # If accuracy weight is zero, return infinite error
    # TODO: remove this
    if np.allclose(w[0], 0, atol=1e-5):
        logging.info('\t\tAccuracy weight is zero, infinite error')
        best_err = np.inf

    return best_err, best_j, mu_deltas, l2_mu_deltas
