import datetime
import itertools
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fairlearn.postprocessing import ThresholdOptimizer
from research.rl.env.clf_mdp import *
from research.rl.env.clf_mdp_policy import *
from research.rl.env.objectives import *
from research.irl.fair_irl import *
from research.ml.svm import SVM
from research.utils import *
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from .datasets import *


# Color palette for plotting
cp = sns.color_palette()


# Objective looup
OBJ_LOOKUP_BY_NAME = {
    'Acc': AccuracyObjective,
    'DemPar': DemographicParityObjective,
    'EqOpp': EqualOpportunityObjective,
    'PredPar': PredictiveEqualityObjective,
}


class FairLearnSkLearnWrapper():
    """
    Wrapper around scikit-learn classifiers to make them compatible with
    fairlearn classifiers, which require an additional `sensitive_features`
    attribute to be passed in when calling `fit` and `predict`.
    """
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


class UnfairNoisyClassifier():
    """
    Wrapper around scikit-learn classifiers to intentionally make them slightly
    fair and less accurate. This is useful when generating initial policies for
    where having reasonably fair and accurate (but not optimal) classifiers
    helps set the weights in the right directions.
    """

    def __init__(self, clf, prob):
        self.clf = clf
        self.prob = prob

    def fit(self, X, y, **kwargs):
        self.clf.fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        preds = self.clf.predict(X, **kwargs)
        override_indexes = np.argwhere(
            np.random.rand(len(X)) < self.prob
        ).flatten()

        for idx in override_indexes:
            if X.iloc[idx]['z'] == 1:
                preds[idx] = 1
            else:
                preds[idx] = 0
        return preds


def generate_expert_algo_lookup(feature_types):
    """
    Parameters
    ----------
    feature_types : dict<str, array-like>
        Mapping of column names to their type of feature. Used to when
        constructing the sklearn pipeline.

    Returns
    -------
    expert_algo_lookup : dict<str, sklearn.pipeline>
        The expert algo lookup dictionary that maps the string name for an
        algorithm to the actual implementation.
    """
    # OptAcc
    opt_acc_pipe = sklearn_clf_pipeline(
        feature_types,
        RandomForestClassifier(),
    )

    # HardtDemPar
    dem_par_thresh_opt = ThresholdOptimizer(
        constraints='demographic_parity',
        predict_method="predict",
        prefit=False,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=RandomForestClassifier(), # Messes up DemPar. Why?
            clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
        )
    )
    dem_par_wrapper = FairLearnSkLearnWrapper(
        clf=dem_par_thresh_opt,
        sensitive_features='z',
    )

    # HardtEqOpp
    eq_opp_thresh_opt = ThresholdOptimizer(
        constraints='true_positive_rate_parity',
        predict_method="predict",
        prefit=False,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
        )
    )
    eq_opp_wrapper = FairLearnSkLearnWrapper(
        clf=eq_opp_thresh_opt,
        sensitive_features='z',
    )

    # PredEq
    pred_eq_thresh_opt = ThresholdOptimizer(
        constraints='false_negative_rate_parity',
        predict_method="predict",
        prefit=False,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
        )
    )
    pred_eq_wrapper = FairLearnSkLearnWrapper(
        clf=pred_eq_thresh_opt,
        sensitive_features='z',
    )

    dummy_pipe = UnfairNoisyClassifier(
        clf=DummyClassifier(strategy="uniform"),
        prob=0.1,
    )

    expert_algo_lookup = {
        'OptAcc': opt_acc_pipe,
        'HardtDemPar': dem_par_wrapper,
        'HardtEqOpp': eq_opp_wrapper,
        'PredEq': pred_eq_wrapper,
        'Dummy': dummy_pipe,
        # Noisy
        'OptAccNoisy': UnfairNoisyClassifier(clf=opt_acc_pipe, prob=.1),
        'HardtDemParNoisy': UnfairNoisyClassifier(clf=dem_par_wrapper, prob=.1),
        'HardtEqOppNoisy': UnfairNoisyClassifier(clf=eq_opp_wrapper, prob=.1),
        'PredEqNoisy': UnfairNoisyClassifier(clf=pred_eq_wrapper, prob=.1),
        'DummyNoisy': UnfairNoisyClassifier(clf=dummy_pipe, prob=.1),
    }

    return expert_algo_lookup


def generate_all_exp_results_df(
        obj_set, n_trials, data_demo, exp_algo, irl_method,
):
    """
    Generate dataframe for experiment parameters and results

    Parameters
    ----------
    obj_set : ObjectiveSet
        The objective set.
    n_trials : int
        Number of experiment trials to run.
    data_demo : str
        The name of the dataset used to generate expert demonstrations.
    exp_algo : str
        The name of the algorithm used to train the expert demonstrator.
    irl_method : str
        The name of the IRL algorithm used to recover the rewards.

    Returns
    -------
    all_exp_df : pandas.DataFrame
        A dataframe with relevant columns, but no data.
    """
    all_exp_df_cols=['n_trials', 'data_demo', 'exp_algo', 'irl_method']

    for obj in obj_set.objectives:
        all_exp_df_cols.append(f"muE_{obj.name}_mean")
        all_exp_df_cols.append(f"muE_{obj.name}_std")

    for obj in obj_set.objectives:
        all_exp_df_cols.append(f"wL_{obj.name}_mean")
        all_exp_df_cols.append(f"wL_{obj.name}_std")

    for obj in obj_set.objectives:
        all_exp_df_cols.append(f"muL_err_{obj.name}_mean")
        all_exp_df_cols.append(f"muL_err_{obj.name}_std")

    all_exp_df_cols.append('muL_err_l2norm_mean')
    all_exp_df_cols.append('muL_err_l2norm_std')

    all_exp_df = pd.DataFrame(columns=all_exp_df_cols)

    all_exp_df.loc[0, 'n_trials'] = n_trials
    all_exp_df.loc[0, 'data_demo'] = 'Adult'
    all_exp_df.loc[0, 'exp_algo'] = exp_algo
    all_exp_df.loc[0, 'irl_method'] = 'IRL_METHOD'

    return all_exp_df


def generate_single_exp_results_df(obj_set, results):
    """
    Generate dataframe for a single experiment. Keeps track of the results of
    the best learned policy.

    Parameters
    ----------
    obj_set : ObjectiveSet
        The objective set.
    data : list<list>
        The results.
    results : pandas.DataFrame
        A dataframe with relevant weight, feat exp, and error columns for the
        best learned policy. Each row is produced by the `new_trial_result()`
        method.
    """
    exp_df_cols = []

    for obj in obj_set.objectives:
        exp_df_cols.append(f"muE_{obj.name}_mean")
        exp_df_cols.append(f"muE_{obj.name}_std")

    for obj in obj_set.objectives:
        exp_df_cols.append(f"muE_hold_{obj.name}_mean")
        exp_df_cols.append(f"muE_hold_{obj.name}_std")

    for obj in obj_set.objectives:
        exp_df_cols.append(f"wL_{obj.name}")

    for obj in obj_set.objectives:
        exp_df_cols.append(f"muL_{obj.name}")
        exp_df_cols.append(f"muL_best_{obj.name}")
        exp_df_cols.append(f"muL_hold_{obj.name}")
        exp_df_cols.append(f"muL_best_hold_{obj.name}")

    for obj in obj_set.objectives:
        exp_df_cols.append(f"muL_err_{obj.name}")
        exp_df_cols.append(f"muL_hold_err_{obj.name}")

    exp_df_cols.append('muL_err_l2norm')
    exp_df_cols.append('muL_hold_err_l2norm')

    for obj in obj_set.objectives:
        exp_df_cols.append(f"muE_target_{obj.name}")
        exp_df_cols.append(f"muL_target_hold_{obj.name}")

    exp_df = pd.DataFrame(results, columns=exp_df_cols)

    return exp_df


def new_trial_result(
        obj_set, muE, muE_hold, df_irl, muE_target=None, muL_target_hold=None,
):
    """
    Generates a row of "results", which are collected and persisted for each
    experiment.

    Parameters
    ---------
    obj_set : research.irl.fair_irl.ObjectiveSet
        The set of objectives.
    muE : array-like<float>, shape(n_expert_demos, n_objectives)
        The feature expectations of the expert demonstrations, specifically on
        the demo set.
    muE : array-like<float>, shape(n_expert_demos, n_objectives)
        The feature expectations of the expert demonstrations on the hold out
        set.
    df_irl : pandas.DataFrame
        A collection of results where each row represents either an expert demo
        (and therefore a positive SVM training example) or a learned policy
        (and therefore a negative SVM training example). Includes relevant
        items like learned weights, feature expectations, error, etc.
    muE_target : array-like<float>. Shape(n_expert_demos, n_objectives).
        The feature expectations of running the expert algo on the target
        domain demo set.
    muL_target_hold
        The feature expectations of running the expert algo on the target
        domain holdout set.

    Returns
    -------
    result : list<list<numeric>>
        The new result row.
    """
    result = []

    for i, obj in enumerate(obj_set.objectives):
        muE_mean = np.mean(muE[:, i])
        muE_std = np.std(muE[:, i])
        result += [muE_mean, muE_std]

    for i, obj in enumerate(obj_set.objectives):
        muE_hold_mean = np.mean(muE_hold[:, i])
        muE_hold_std = np.std(muE_hold[:, i])
        result += [muE_hold_mean, muE_hold_std]

    best_t = (
        df_irl.query('(is_expert == 0) and (is_init_policy == 0)')
        .sort_values('t')
        ['t'].values[0]
    )
    best_idx = (
        df_irl[
            (df_irl['is_expert'] == 0) &
            (df_irl['is_init_policy'] == 0)
        ]
        .query('abs(t - @best_t) <= .0001')
        .sort_index()
        .index[0]
    )
    best_row = df_irl.loc[best_idx]

    for i, obj in enumerate(obj_set.objectives):
        result.append(best_row[f"{obj.name}_weight"])

    for obj in obj_set.objectives:
        result.append(best_row[f"{obj.name}"])
        result.append(best_row[f"muL_best_{obj.name}"])
        result.append(best_row[f"muL_hold_{obj.name}"])
        result.append(best_row[f"muL_best_hold_{obj.name}"])

    for i, obj in enumerate(obj_set.objectives):
        _muL_err = abs(best_row[f"{obj.name}"] - np.mean(muE_hold[:, i]))
        _muL_hold_err = abs(
            best_row[f"muL_hold_{obj.name}"] - np.mean(muE_hold[:, i])
        )
        result.append(_muL_err)
        result.append(_muL_hold_err)

    result.append(best_row['mu_delta_l2norm'])
    result.append(best_row['mu_delta_l2norm_hold'])

    for i, obj in enumerate(obj_set.objectives):

        if muE_target is not None:
            result.append(muE_target[i])
        else:
            result.append(np.nan)

        if muL_target_hold is not None:
            result.append(muL_target_hold[i])
        else:
            result.append(np.nan)

    return result


def run_trial_source_domain(
    exp_info, X=None, y=None, feature_types=None, plot_svm_iters=False,
):
    """
    Runs 1 trial to learn weights in the source domain.

    X, y, feature_types don't need to be passed. If they are, then
    `generate_dataset()` is not invoked.

    Parameters
    ----------
    exp_info : dict
        Metadata about the experiment.
    X : pandas.DataFrame, Optional
        The X (including z) columns.
    y : pandas.Series, Optional
        Just the y column.
    feature_types : dict<str, array-like>, Optional
        Mapping of column names to their type of feature. Used to when
        constructing sklearn pipelines.
    plot_svm_iters : bool, default False
        If True, plots the SVM iterations.

    Returns
    -------
    muE : array-like<float>. Shape(n_expert_demos, n_objectives)
        Expert demonstration feature expectations.
    muE : array-like<float>. Shape(n_expert_demos, n_objectives)
        Expert feature expectations on the hold out set.
    df_irl : pandas.DataFrame
        A collection of results where each row represents either an expert demo
        (and therefore a positive SVM training example) or a learned policy
        (and therefore a negative SVM training example). Includes relevant
        items like learned weights, feature expectations, error, etc.
    weights : array-like<float>. Shape(n_irl_loop_iters,  n_objectives).
        The learned weights for each iteration of the IRL loop.
    t_hold : array-like<float>. Shape(n_irl_loop_iters)
        The irl error on the holdout set.
    """
    # Initiate objectives
    objectives = []
    for obj_name in exp_info['OBJECTIVE_NAMES']:
        objectives.append(OBJ_LOOKUP_BY_NAME[obj_name]())
    obj_set = ObjectiveSet(objectives)
    del objectives

    # Set observability level
    if 'FO' in exp_info['IRL_METHOD']:
        CAN_OBSERVE_Y = True
    else:
        CAN_OBSERVE_Y = False

    # Reset the objective set since they get fitted in each trial run
    obj_set.reset()

    # Read in dataset
    if X is None or y is None or feature_types is None:
        X, y, feature_types = generate_dataset(
            exp_info['DATASET'],
            n_samples=exp_info['N_DATASET_SAMPLES'],
        )

    # These are the feature type sthat will be used as inputs for the expert
    # classifier.
    expert_demo_feature_types = feature_types

    # These are the feature types that will be used in the classifier that will
    # predict `y` given `X` when learning the optimal policy for a given reward
    # function.
    irl_loop_feature_types = feature_types

    expert_algo_lookup = generate_expert_algo_lookup(expert_demo_feature_types)

    # Split data into 3 sets.
    #     1. Demo - Produces expert demonstratinos
    #         1A. Train – Used for predicting Y from Z,X
    #         1B. Test – used for comparing muL with muE
    #     2. Hold – computes the unbiased values for muL and t (dataset is
    #.       never shown to the IRL learning algo.)
    X_demo, X_hold, y_demo, y_hold = train_test_split(X, y, test_size=.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X_demo,
        y_demo,
        test_size=.5,
    )
    del X_demo, y_demo # Make sure I don't acidentally use these later on

    # Generate expert demonstrations to learn from
    muE, demosE = generate_demos_k_folds(
        X=X_test,
        y=y_test,
        clf=expert_algo_lookup[exp_info['EXPERT_ALGO']],
        obj_set=obj_set,
        n_demos=exp_info['N_EXPERT_DEMOS'],
    )
    logging.info(f"muE:\n{muE}")

    # Generate expert demonstrations to learn for computing learned
    # performance. These expert demos are never shown to the IRL algo and are
    # only used for performance measurement.
    muE_hold, demosE_hold = generate_demos_k_folds(
        X=X_hold,
        y=y_hold,
        clf=expert_algo_lookup[exp_info['EXPERT_ALGO']],
        obj_set=obj_set,
        n_demos=exp_info['N_EXPERT_DEMOS'],
    )
    logging.info(f"muE_hold:\n{muE_hold}")

    ##
    # Run IRL loop.
    # Create a clf dataset where inputs are feature expectations and outputs
    # are whether the policy is expert or learned through IRL iterations. Then
    # train an SVM classifier on this dataset. Then extract the weights of the
    # svm and use them as the weights for the "reward" function. Then use this
    # reward function to learn a policy (classifier). Then compute the feature
    # expectations from this classifer on the irl hold-out set. Then compute
    # the error between the feature expectations of this learned clf and the
    # demonstration feature exp. If this error is less than epsilon, stop. The
    # reward function is the final set of weights.
    ##

    # Initiate variables needed to run IRL Loop
    x_cols = (
        irl_loop_feature_types['boolean']
        + irl_loop_feature_types['categoric']
        + irl_loop_feature_types['continuous']
    )
    x_cols.remove('z')
    obj_set_cols = [obj.name for obj in obj_set.objectives]

    # Generate initial learned policies to serve as negative training examples
    # for the SVM IRL classifier.
    mu = []
    for non_expert_algo in exp_info['NON_EXPERT_ALGOS']:
        _mu, _demos = generate_demos_k_folds(
            X=X_train,
            y=y_train,
            clf=expert_algo_lookup[non_expert_algo],
            obj_set=obj_set,
            n_demos=1,
        )
        mu.append(_mu[0])

    mu = np.array(mu)
    logging.info(f"muL:\n{mu}")
    X_irl_exp = pd.DataFrame(muE, columns=obj_set_cols)
    y_irl_exp = pd.Series(np.ones(exp_info['N_EXPERT_DEMOS']), dtype=int)
    X_irl_learn = pd.DataFrame(mu, columns=obj_set_cols)
    y_irl_learn = pd.Series(np.zeros(len(mu)), dtype=int)

    t = []  # Errors for each iteration
    t_hold = []  # Errors on hold out set for each iteration
    mu_delta_l2norm_hist = []
    mu_delta_l2norm_hold_hist = []
    weights = []
    i = 0
    demo_history = []
    demo_hold_history = []
    mu_history = []
    mu_hold_history = []
    mu_best_history = []
    mu_best_hold_history = []

    # Start the IRL Loop
    logging.debug('')
    logging.debug('Starting IRL Loop ...')

    while True:
        logging.info(f"\tIRL Loop iteration {i+1}/{exp_info['MAX_ITER']} ...")

        # Train SVM classifier that distinguishes which demonstrations are
        # expert and which were generated from this loop.
        logging.debug('\tFitting SVM classifier...')
        X_irl = (
            pd.concat([X_irl_exp, X_irl_learn], axis=0)
            .reset_index(drop=True)
        )
        y_irl = (
            pd.concat([y_irl_exp, y_irl_learn], axis=0)
            .reset_index(drop=True)
        )
        svm = SVM(positive_weights_only=True).fit(X_irl, y_irl)
        wi = svm.weights()
        weights.append(wi)

        ##
        # Learn a policy (clf_pol) from the reward (SVM) weights.
        ##

        # Fit a classifier that predicts `y` from `X`.
        logging.debug('\tFitting `y|x` predictor for clf policy...')
        clf = sklearn_clf_pipeline(
            feature_types=irl_loop_feature_types,
            clf_inst=RandomForestClassifier(),
        )
        clf.fit(X_train, y_train)

        # Learn a policy that maximizes the reward function.
        logging.debug('\tComputing the optimal policy given reward weights and `y|x` classifier...')
        reward_weights = { obj.name: wi[j] for j, obj in enumerate(obj_set.objectives) }
        test_df = pd.DataFrame(X_test)
        #
        # JDB 06/27/2023 – NOTE: mu0s need to be calculated using the predicted
        # labels, not the true labels. Otherwise the optimizer solves for a
        # policy that doesn't reflect observability. I discovered this issue by
        # giving weights [0, 1, 0] into the run_trial_source_domain and
        # noticing that the ouptput had poor demographic parity.
        #
        # test_df['y'] = y_test
        test_df['y'] = clf.predict(X_test)
        clf_pol = compute_optimal_policy(
            clf_df=test_df,  # NOT the dataset used to train the C_{Y_Z,X} clf
            clf=clf,
            x_cols=x_cols,
            obj_set=obj_set,
            reward_weights=reward_weights,
            skip_error_terms=True,
            method=exp_info['METHOD'],
            min_freq_fill_pct=exp_info['MIN_FREQ_FILL_PCT'],
        )

        ##
        # Measure and record the error of the learned policy, and keep it as
        # a negative training example for next IRL Loop iteration.
        ##

        # Compute feature expectations of the learned policy
        logging.debug('\tGenerating learned demostration...')
        demo = generate_demo(clf_pol, X_test, y_test, can_observe_y=CAN_OBSERVE_Y)
        demo_hold = generate_demo(clf_pol, X_hold, y_hold, can_observe_y=False)
        demo_history.append(demo)
        demo_hold_history.append(demo_hold)
        muj = obj_set.compute_demo_feature_exp(demo)
        muj_hold = obj_set.compute_demo_feature_exp(demo_hold)
        mu_history.append(muj)
        mu_hold_history.append(muj_hold)
        logging.info(f"\t\t muL[{i}] = {np.round(muj, 3)}")
        logging.debug(f"\t\t muL_hold[{i}] = {np.round(muj_hold, 3)}")

        # Append policy's feature expectations to irl clf dataset
        X_irl_learn_i = pd.DataFrame(np.array([muj]), columns=obj_set_cols)
        y_irl_learn_i = pd.Series(np.zeros(1), dtype=int)
        X_irl_learn = pd.concat([X_irl_learn, X_irl_learn_i], axis=0)
        y_irl_learn = pd.concat([y_irl_learn, y_irl_learn_i], axis=0)

        # Compute error of the learned policy: t[i] = wT(muE-mu[j])
        # This is equivalent to computing the SVM margin.
        ti, best_j, mu_delta, mu_delta_l2norm = irl_error(
            wi,
            muE,
            mu_history,
            norm_weights=exp_info['IRL_ERROR_NORM_WEIGHTS'],
        )
        # Do it for the hold-out set as well.
        ti_hold, best_j_hold, mu_delta_hold, mu_delta_l2norm_hold = irl_error(
            wi,
            muE_hold,
            mu_hold_history,
            norm_weights=exp_info['IRL_ERROR_NORM_WEIGHTS'],
        )
        mu_best_history.append(mu_history[best_j])
        mu_best_hold_history.append(mu_hold_history[best_j])
        t.append(ti)
        t_hold.append(ti_hold)
        mu_delta_l2norm_hist.append(mu_delta_l2norm)
        mu_delta_l2norm_hold_hist.append(mu_delta_l2norm_hold)
        logging.info(f"\t\t mu_delta[{i}] \t= {np.round(mu_delta, 3)}")
        logging.debug(f"\t\t mu_delta_hold[i] \t= {np.round(mu_delta_hold, 3)}")
        logging.info(f"\t\t t[{i}] \t\t= {t[i]:.5f}")
        logging.debug(f"\t\t t_hold[i] \t\t= {t_hold[i]:.5f}")
        logging.info(f"\t\t weights[{i}] \t= {np.round(weights[i], 3)}")

        ## Show a summary of the learned policy
        # logging.info(
        #     df_to_log(
        #         (
        #             demo.groupby(['z']+x_cols+['y', 'yhat'])
        #             [['age']].agg(['count'])
        #         ),
        #         title='\tLearned Policy:',
        #         tab_level=3,
        #     )
        # )

        if ti < exp_info['EPSILON'] or i >= exp_info['MAX_ITER'] - 1:
            break

        i += 1

        # End IRL Loop

    ##
    # Book keeping stuff for the trial.
    ##

    # Compare the best learned policy with the expert demonstrations
    best_t = np.min(t)
    best_iter = None
    for t_i, _t in enumerate(t):
        if np.allclose(best_t, _t, atol=.0001):
            best_iter = t_i
            break

    best_demo = demo_history[best_iter]
    best_weight = weights[best_iter]
    logging.debug('Best iteration: ' + str(best_iter))
    logging.info(f"Best Learned Policy yhat: {best_demo['yhat'].mean():.3f}")
    logging.info(f"best weight:\t {np.round(best_weight, 3)}")

    # Generate a dataframe for results gathering.
    X_irl = pd.concat([X_irl_exp, X_irl_learn], axis=0).reset_index(drop=True)
    y_irl = pd.concat([y_irl_exp, y_irl_learn], axis=0).reset_index(drop=True)
    df_irl = X_irl.copy()
    df_irl['is_expert'] = y_irl.copy()
    for i, col in enumerate(obj_set_cols):
        df_irl[f"muL_best_{col}"] = (
            np.zeros(
                exp_info['N_EXPERT_DEMOS'] + exp_info['N_INIT_POLICIES']
            ).tolist() + np.array(mu_best_history)[:, i].tolist()
        )
        df_irl[f"muL_hold_{col}"] = (
            np.zeros(
                exp_info['N_EXPERT_DEMOS'] + +exp_info['N_INIT_POLICIES']
            ).tolist() + np.array(mu_hold_history)[:, i].tolist()
        )
        df_irl[f"muL_best_hold_{col}"] = (
            np.zeros(
                exp_info['N_EXPERT_DEMOS'] + exp_info['N_INIT_POLICIES']
            ).tolist() + np.array(mu_best_hold_history)[:, i].tolist()
        )
    df_irl['is_init_policy'] = (
        np.zeros(exp_info['N_EXPERT_DEMOS']).tolist()
        + np.ones(exp_info['N_INIT_POLICIES']).tolist()
        + np.zeros(len(t)).tolist()
    )
    df_irl['learn_idx'] = (
        list(-1*np.ones(exp_info['N_EXPERT_DEMOS']))
        + list(np.arange(exp_info['N_INIT_POLICIES'] + len(t)))
    )
    for i, col in enumerate(obj_set_cols):
        df_irl[f"{col}_weight"] = (
            np.zeros(
                exp_info['N_EXPERT_DEMOS']+exp_info['N_INIT_POLICIES']
            ).tolist() + [w[i] for w in weights]
        )
    df_irl['t'] = (
        list(
            np.inf*(
                np.ones(exp_info['N_EXPERT_DEMOS']+exp_info['N_INIT_POLICIES'])
            )
        ) + t
    )
    df_irl['t_hold'] = (
        list(
            np.inf*(
                np.ones(exp_info['N_EXPERT_DEMOS']+exp_info['N_INIT_POLICIES'])
            )
        ) + t_hold
    )
    df_irl['mu_delta_l2norm'] = (
        np.zeros(exp_info['N_EXPERT_DEMOS']+exp_info['N_INIT_POLICIES'])
        .tolist() + mu_delta_l2norm_hist
    )
    df_irl['mu_delta_l2norm_hold'] = (
        np.zeros(exp_info['N_EXPERT_DEMOS']+exp_info['N_INIT_POLICIES'])
        .tolist() + mu_delta_l2norm_hold_hist
    )
    logging.debug('Experiment Summary')
    display(df_irl.round(3))

    ###
    ## Plot results
    ###
    if plot_svm_iters:
        sns.set_theme(style='darkgrid', font_scale=1.2)
        feat_exp_combs = list(itertools.combinations(obj_set_cols, 2))
        exp = df_irl.query('is_expert == True').reset_index(drop=True)
        lrn = df_irl.query('is_expert == False').reset_index(drop=True)
        best_t_idx = lrn.query('t > 0')['t'].argmin()
        fig, axes = plt.subplots(
            1,
            len(feat_exp_combs),
            figsize=(5*len(feat_exp_combs), 4),
        )
        axes = (axes,) if len(feat_exp_combs) == 1 else axes
        for i, (feat_exp_x, feat_exp_y) in enumerate(feat_exp_combs):
            # Plot expert
            axes[i].scatter(
                exp[feat_exp_x],
                exp[feat_exp_y],
                label='$\mu^E$',
                s=600,
                alpha=1,
                c='black',
            )
            # Inject noise so we can see the expert when it's overlapping
            noise = exp_info['NOISE_FACTOR']*(np.random.rand(len(lrn))-.6)
            # Plot the learned policies
            axes[i].scatter(
                lrn[feat_exp_x]+noise,
                lrn[feat_exp_y]+noise,
                label='$\mu^L_i$',
                s=600,
                alpha=.7,
                c=cp[2],
            )
            axes[i].set_ylim([-.1, 1.1])
            axes[i].set_xlim([-.1, 1.1])
            axes[i].set_xlabel(feat_exp_x.replace('_', ' ').title())
            axes[i].set_ylabel(feat_exp_y.replace('_', ' ').title())
            if exp_info['ANNOTATE']:
                # Label each learned policy with its ordered index
                for idx, row in lrn.iterrows():
                    if row['is_init_policy']:
                        annotation = None
                    else:
                        annotation = idx
                    axes[i].annotate(
                        annotation,
                        (
                            -.012+(row[feat_exp_x]+noise[idx]),
                            -.015+(row[feat_exp_y]+noise[idx])
                        ),
                        fontsize=16,
                        fontweight=700,
                    )
            # Color the best policy
            axes[i].scatter(
                [lrn.loc[best_t_idx][feat_exp_x]+noise[best_t_idx]],
                [lrn.loc[best_t_idx][feat_exp_y]+noise[best_t_idx]],
                label='Best $\mu^L_i$',
                s=600,
                alpha=1,
                c=cp[1],
            )
            axes[i].legend(ncol=1, labelspacing=.7, loc='upper left')

        plt.suptitle(f"Best learned weights: {best_weight.round(2)}")
        plt.tight_layout()

    # End regular IRL trial

    return muE, muE_hold, df_irl, weights, t_hold


def run_trial_target_domain(
        exp_info, weights, t_hold, X=None, y=None, feature_types=None,
):
    """
    Runs 1 trial to learn a policy in a new domain using weights learned from
    another domain.

    X, y, feature_types don't need to be passed. If they are, then
    `generate_dataset()` is not invoked.

    Parameters
    ----------
    exp_info : dict
        Metadata about the experiment.
    weights : array-like<float>, shape (n_trials, n_objectives)
        The learned weights from the source domain.
    t_hold : array-like<float>, len(n_trials)
        The irl error histories. This is how we are selecting which weights to
        in the target domain.
    X : pandas.DataFrame, Optional
        The X (including z) columns.
    y : pandas.Series, Optional
        Just the y column.
    feature_types : dict<str, array-like>, Optional
        Mapping of column names to their type of feature. Used to when
        constructing sklearn pipelines.

    Returns
    -------
    muE_target : array-like<float>. Shape(n_expert_demos, n_objectives).
        The feature expectations of running the expert algo on the target
        domain demo set.
    muL_target_hold
        The feature expectations of running the expert algo on the target
        domain holdout set.
    """
    if 'FO' in exp_info['IRL_METHOD']:
        CAN_OBSERVE_Y = True
    else:
        CAN_OBSERVE_Y = False

    objectives = []
    for obj_name in exp_info['OBJECTIVE_NAMES']:
        objectives.append(OBJ_LOOKUP_BY_NAME[obj_name]())
    obj_set = ObjectiveSet(objectives)
    del objectives
    obj_set_cols = [obj.name for obj in obj_set.objectives]

    # Get the best weights from the source domain
    source_best_w = np.zeros(len(obj_set_cols))
    for i, o in enumerate(obj_set_cols):
        best_idx = np.argmin(t_hold)
        source_best_w[i] = weights[best_idx][i]

    # Read in target domain dataset
    if X is None or y is None or feature_types is None:
        X, y, feature_types = generate_dataset(
            exp_info['TARGET_DATASET'],
            n_samples=exp_info['N_DATASET_SAMPLES'],
        )

    x_cols = (
        feature_types['boolean']
        + feature_types['categoric']
        + feature_types['continuous']
    )
    x_cols.remove('z')

    # These are the feature type sthat will be used as inputs for the expert
    # classifier.
    expert_demo_feature_types = feature_types

    # These are the feature types that will be used in the classifier that will
    # predict `y` given `X` when learning the optimal policy for a given reward
    # function.
    irl_loop_feature_types = feature_types

    expert_algo_lookup = generate_expert_algo_lookup(expert_demo_feature_types)

    ##
    # Split data into 3 sets.
    #     1. Demo - Produces expert demonstratinos
    #         1A. Train – Used for predicting Y from Z,X
    #         1B. Test – used for comparing muL with muE
    #     2. Hold – computes the unbiased values for muL and t (dataset is
    #.       never shown to the IRL learning algo.)
    ##
    X_demo, X_hold, y_demo, y_hold = train_test_split(X, y, test_size=.5)
    X_train, X_test, y_train, y_test = train_test_split(
        X_demo,
        y_demo,
        test_size=.5,
    )
    del X_demo, y_demo # Make sure I don't acidentally use these vars later on

    # Generate expert demonstrations to compare against.
    muE_target, demosE_target = generate_demos_k_folds(
        X=X_test,
        y=y_test,
        clf=expert_algo_lookup[exp_info['EXPERT_ALGO']],
        obj_set=obj_set,
        n_demos=exp_info['N_EXPERT_DEMOS'],
    )
    logging.info(f"muE_target:\n{muE_target}")

    ##
    # Learn a policy (clf_pol) from the reward (SVM) weights.
    ##

    # Fit a classifier that predicts `y` from `X`.
    clf = sklearn_clf_pipeline(
        feature_types=feature_types,
        clf_inst=RandomForestClassifier(),
    )
    clf.fit(X_train, y_train)

    # Learn a policy that maximizes the reward function.
    reward_weights = {
        obj.name: source_best_w[j] for j, obj in enumerate(obj_set.objectives)
    }
    test_df = pd.DataFrame(X_test)
    #
    # JDB 06/27/2023 – NOTE: mu0s need to be calculated using the predicted
    # labels, not the true labels. Otherwise the optimizer solves for a
    # policy that doesn't reflect observability. I discovered this issue by
    # giving weights [0, 1, 0] into the run_trial_source_domain and
    # noticing that the ouptput had poor demographic parity.
    #
    # test_df['y'] = y_test
    test_df['y'] = clf.predict(X_test)
    clf_pol = compute_optimal_policy(
        clf_df=test_df,  # NOT the dataset used to train the C_{Y_Z,X} clf
        clf=clf,
        x_cols=x_cols,
        obj_set=obj_set,
        reward_weights=reward_weights,
        skip_error_terms=True,
        method=exp_info['METHOD'],
        min_freq_fill_pct=exp_info['MIN_FREQ_FILL_PCT'],
    )

    # Compute feature expectations of the learned policy
    demo = generate_demo(clf_pol, X_test, y_test, can_observe_y=CAN_OBSERVE_Y)
    demo_hold = generate_demo(clf_pol, X_hold, y_hold, can_observe_y=False)
    muL_target = obj_set.compute_demo_feature_exp(demo)
    muL_target_hold = obj_set.compute_demo_feature_exp(demo_hold)
    logging.info(f"target domain muL = {np.round(muL_target, 3)}")
    logging.info(f"target domain muE = {np.round(muE_target.mean(axis=0), 3)}")
    logging.info(f"target domain muL_hold = {np.round(muL_target_hold, 3)}")

    return muE_target, muL_target_hold, clf_pol


def run_experiment(
        exp_info, source_X=None, source_y=None, source_feature_types=None,
        target_X=None, target_y=None, target_feature_types=None,
):
    """
    Runs experiment for source domain and optionally target domain based on
    the parameters in `exp_info`.

    Parameters
    ----------
    exp_info : dict
        Experiment parameters.

    Returns
    -------
    None

    Persists
    --------
    exp_df : pandas.DataFrame
        Saves experiment results as a CSV where each row in the CSV represents
        the relevant results of one trial. The file is stored as
        "/data/experiment_output/fair_irl/exp_results/{timestamp}.csv"
    exp_info : dict
        Saves the experiment parameters and metadata metadata as a JSON file
        "./../../data/experiment_output/fair_irl/exp_info/{timestamp}.json"
    source_X : pandas.DataFrame, Optional
        The X (including z) columns for the source domain.
    source_y : pandas.Series, Optional
        Just the y column for the source domain.
    source_feature_types : dict<str, array-like>, Optional
        Mapping of column names to their type of feature. Used to when
        constructing sklearn pipelines for the source domain.
    target_X : pandas.DataFrame, Optional
        The X (including z) columns for the target domain.
    target_y : pandas.Series, Optional
        Just the y column for the target domain.
    target_feature_types : dict<str, array-like>, Optional
        Mapping of column names to their type of feature. Used to when
        constructing sklearn pipelines for the target domain.
    """
    logging.info(f"exp_info: {exp_info}")

    objectives = []
    for obj_name in exp_info['OBJECTIVE_NAMES']:
        objectives.append(OBJ_LOOKUP_BY_NAME[obj_name]())
    obj_set = ObjectiveSet(objectives)
    del objectives

    results = []
    trial_i = 0
    while trial_i < exp_info['N_TRIALS']:
        logging.info(f"\n\nTRIAL {trial_i}\n")

        # Run trials to learn weights on source domain
        muE, muE_hold, df_irl, weights, t_hold = run_trial_source_domain(
            exp_info,
            X=source_X,
            y=source_y,
            feature_types=source_feature_types,
        )

        # Learn clf in target domain using weights learned in source domain
        muE_target_mean = None
        muL_target_hold = None

        if exp_info['TARGET_DATASET'] is not None:
            muE_target, muL_target_hold, clf_pol = run_trial_target_domain(
                exp_info,
                weights,
                t_hold,
                X=target_X,
                y=target_y,
                feature_types=target_feature_types,
            )
            muE_target_mean = muE_target.mean(axis=0)

        # Aggregate trial results
        _result = new_trial_result(
            obj_set,
            muE,
            muE_hold,
            df_irl,
            muE_target_mean,
            muL_target_hold,
        )
        results.append(_result)

        trial_i += 1

    # Persist trial results
    exp_df = generate_single_exp_results_df(obj_set, results)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    exp_df.to_csv(
        f"./../../data/experiment_output/fair_irl/exp_results/{timestamp}.csv",
        index=None,
    )

    # Persist trial info
    exp_info['timestamp'] = timestamp
    fp = f"./../../data/experiment_output/fair_irl/exp_info/{timestamp}.json"
    json.dump(exp_info, open(fp, 'w'))

    return None


def plot_results_source_domain_only(
    objective_set_names,
    expert_algo, dataset,
    mu_noise_factor=0,
    w_noise_factor=0,
    mu_hue_order=['muE', 'muL (FairIRL)', 'muL (FairIRLFO)'],
    w_hue_order=[None, 'wL (FairIRL)', 'wL (FairIRLFO)'],
    irl_methods_to_exclude=[
        'FairIRLDeNormed',
        'FairIRLFODeNormed',
        'FairIRLNorm2',
    ],
    extra_skip_conditions=(lambda info: False),
    extra_title=None,
    min_exp_timestamp=None,
    max_exp_timestamp=None,
):
    # Construct a pivot table so we can do a seaborn boxplot
    mu_cols = ['Value', 'Demo Producer', 'Feature Expectation']
    mu_rows = []
    w_cols = ['Value', 'IRL Method', 'Weight']
    w_rows = []

    path_prefix = './../../data/experiment_output/fair_irl/'
    exp_results_files = sorted(os.listdir(f"{path_prefix}/exp_results/"))
    exp_info_files = sorted(os.listdir(f"{path_prefix}/exp_info/"))

    # For each experiment...
    for (result_file, info_file) in zip(exp_results_files, exp_info_files):
        if result_file.replace('csv', '') != info_file.replace('json', ''):
            raise ValueError(
                f"Mismatched number of results and info files. "
                f"{result_file}, {info_file}"
            )

        # Filter out any unwanted experiments
        if min_exp_timestamp is not None and info_file < min_exp_timestamp:
            continue
        if max_exp_timestamp is not None and info_file > max_exp_timestamp:
            continue

        result = pd.read_csv(
            f"{path_prefix}/exp_results/{result_file}",
            index_col=None,
        )
        info = json.load(open(f"{path_prefix}/exp_info/{info_file}"))

        # Filter to only experiments for the input `expert_algo`
        if (
            info['EXPERT_ALGO'] != expert_algo
            or info['DATASET'] != dataset
            or set(info['OBJECTIVE_NAMES']) != set(objective_set_names)
            or info['IRL_METHOD'] in irl_methods_to_exclude
            or extra_skip_conditions(info)
        ):
            continue

        for idx, row in result.iterrows():
            # Append muE and muL results
            for obj_name in info['OBJECTIVE_NAMES']:
                mu_rows.append([
                    row[f"muE_hold_{obj_name}_mean"],
                    'muE',
                    obj_name,
                ])
            for obj_name in info['OBJECTIVE_NAMES']:
                mu_rows.append([
                    row[f"muL_best_hold_{obj_name}"],
                    f"muL ({info['IRL_METHOD']})",
                    obj_name,
                ])
                w_rows.append([
                    row[f"wL_{obj_name}"],
                    f"wL ({info['IRL_METHOD']})",
                    obj_name,
                ])

    if len(mu_rows) == 0:
        raise ValueError(f"No experimets with EXPERT_ALGO={expert_algo}")

    mu_df = pd.DataFrame(mu_rows, columns=mu_cols)
    mu_df['Value'] += mu_noise_factor*(np.random.rand(len(mu_df)) - .5)
    mu_df['Value'] = mu_df['Value'].clip(0, 1)
    w_df = pd.DataFrame(w_rows, columns=w_cols)
    w_df['Value'] += w_noise_factor*(np.random.rand(len(w_df)) - .5)

    mu_df['Demo Producer'] = (
        mu_df['Demo Producer'].str.replace('muE', r'$\\mu^E$')
    )
    mu_df['Demo Producer'] = (
        mu_df['Demo Producer'].str.replace('muL', r'$\\mu^L$')
    )
    w_df['IRL Method'] = (
        w_df['IRL Method'].str.replace('wL', r'$w^L$')
    )

    mu_hue_order = pd.Series(mu_hue_order).str.replace('muE', r'$\\mu^E$')
    mu_hue_order = pd.Series(mu_hue_order).str.replace('muL', r'$\\mu^L$')
    w_hue_order = pd.Series(w_hue_order).str.replace('wL', r'$w^L$')

    # Plot boxplot for feature expectations
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(5, 8.5),
        height_ratios=[2,1.33],
    )
    sns.boxplot(
        x=mu_df['Feature Expectation'],
        y=mu_df['Value'],
        hue=mu_df['Demo Producer'],
        hue_order=mu_hue_order,
        ax=ax1,
        fliersize=0,  # Remove outliers
        saturation=1,
        palette=list(cp[0:3])+ list(cp[6:]),
    )
    ax1.set_ylabel(None)
    ax1.set_xlabel('Learned Feature Expectations')
    ax1.legend(
        title=None,
        # fontsize=18,
        loc='lower left',
    )
    ax1.set_ylim([-.1,1.1])

    # Plot boxplot for weights
    sns.boxplot(
        x=w_df['Weight'],
        y=w_df['Value'],
        hue=w_df['IRL Method'],
        hue_order=w_hue_order,
        ax=ax2,
        fliersize=0,  # Remove outliers
        saturation=1,
        palette=list(cp[0:3])+ list(cp[6:]),
    )
    ax2.set_ylabel(None)
    ax2.set_xlabel('Learned Weights')
    ax2.set_ylim([-1, 1])
    ax2.legend(
        title=None,
        loc='lower left',
        # fontsize=18,
    )

    # Set title on top of fig 1
    title = f"Expert Algo: {expert_algo}"
    if extra_title is not None:
        title += f"\n{extra_title}"
    title += f"\n{28*'-'}"
    ax1.set_title(title)

    plt.tight_layout()

    print(f"DATASET: {dataset}")

    return mu_df, w_df


def plot_results_target_domain(
    objective_set_names, expert_algo, source_dataset, target_dataset,
    mu_noise_factor=0,
    w_noise_factor=0,
    mu_hue_order=[
        'muE',
        'muL (FairIRL)',
        'muL (FairIRLFO)',
        'muE_target',
        'muL_target',
    ],
    w_hue_order=[
        None,
        'wL (FairIRL)',
        'wL (FairIRLFO)',
        # 'wL (FairIRLNorm)',
        # 'wL (FairIRLNorm2)'
    ],
    irl_methods_to_exclude=[
        'FairIRLDeNormed',
        'FairIRLFODeNormed',
        'FairIRLNorm2',
    ],
    extra_skip_conditions=(lambda info: False),
    min_exp_timestamp=None,
    max_exp_timestamp=None,
):
    # Construct a pivot table so we can do a seaborn boxplot
    mu_cols = ['Value', 'Demo Producer', 'Feature Expectation']
    mu_rows = []
    w_cols = ['Value', 'IRL Method', 'Weight']
    w_rows = []

    path_prefix = './../../data/experiment_output/fair_irl/'
    exp_results_files = sorted(os.listdir(f"{path_prefix}/exp_results/"))
    exp_info_files = sorted(os.listdir(f"{path_prefix}/exp_info/"))

    # For each experiment...
    for (result_file, info_file) in zip(exp_results_files, exp_info_files):
        if result_file.replace('csv', '') != info_file.replace('json', ''):
            raise ValueError(
                f"Mismatched number of results and info files. "
                f"{result_file}, {info_file}"
            )

        # Filter out any unwanted experiments
        if min_exp_timestamp is not None and info_file < min_exp_timestamp:
            continue
        if max_exp_timestamp is not None and info_file > max_exp_timestamp:
            continue

        result = pd.read_csv(
            f"{path_prefix}/exp_results/{result_file}",
            index_col=None,
        )
        info = json.load(open(f"{path_prefix}/exp_info/{info_file}"))

        # Filter to only experiments for the input `expert_algo`
        if (
            info['EXPERT_ALGO'] != expert_algo
            or info['DATASET'] != source_dataset
            or info['TARGET_DATASET'] != target_dataset
            or set(info['OBJECTIVE_NAMES']) != set(objective_set_names)
            or info['IRL_METHOD'] in irl_methods_to_exclude
            or extra_skip_conditions(info)
        ):
            continue

        for idx, row in result.iterrows():
            # Append muE and muL results
            for obj_name in info['OBJECTIVE_NAMES']:
                mu_rows.append([
                    row[f"muE_hold_{obj_name}_mean"],
                    'muE',
                    obj_name,
                ])
                if f"muE_target_{obj_name}" in row:
                    mu_rows.append([
                        row[f"muE_target_{obj_name}"],
                        'muE_target',
                        obj_name,
                    ])
                    mu_rows.append([
                        row[f"muL_target_hold_{obj_name}"],
                        'muL_target' + ' ' + info['IRL_METHOD'],
                        obj_name,
                    ])
            for obj_name in info['OBJECTIVE_NAMES']:
                mu_rows.append([
                    row[f"muL_best_hold_{obj_name}"],
                    f"muL ({info['IRL_METHOD']})",
                    obj_name,
                ])
                w_rows.append([
                    row[f"wL_{obj_name}"],
                    f"wL ({info['IRL_METHOD']})",
                    obj_name,
                ])

    if len(mu_rows) == 0:
        raise ValueError(f"No experimets with EXPERT_ALGO={expert_algo}")

    mu_df = pd.DataFrame(mu_rows, columns=mu_cols)
    mu_df['Value'] += mu_noise_factor*(np.random.rand(len(mu_df)) - .5)
    mu_df['Value'] = mu_df['Value'].clip(0, 1)
    mu_df = mu_df[
        mu_df['Demo Producer'].str.contains('muE')
        | mu_df['Demo Producer'].str.contains('muL_target')
    ]


    w_df = pd.DataFrame(w_rows, columns=w_cols)
    w_df['Value'] += w_noise_factor*(np.random.rand(len(w_df)) - .5)
    mu_df['Demo Producer'] = (
        mu_df['Demo Producer'].str.replace('muE_target', r'$\\mu^E_{TARGET}$')
    )
    mu_df['Demo Producer'] = (
        mu_df['Demo Producer'].str.replace('muE', r'$\\mu^E_{SOURCE}$')
    )
    mu_df['Demo Producer'] = (
        mu_df['Demo Producer'].str.replace('muL_target', r'$\\mu^L_{TARGET}$')
    )
    mu_df['Demo Producer'] = (
        mu_df['Demo Producer'].str.replace('muL', r'$\\mu^L$')
    )
    mu_hue_order = (
        pd.Series(mu_hue_order).str.replace('muE_target', r'$\\mu^E_{TARGET}$')
    )
    mu_hue_order = (
        pd.Series(mu_hue_order).str.replace('muE', r'$\\mu^E_{SOURCE}$')
    )
    mu_hue_order = (
        pd.Series(mu_hue_order).str.replace('muL_target', r'$\\mu^L_{TARGET}$')
    )
    mu_hue_order = (
        pd.Series(mu_hue_order).str.replace('muL', r'$\\mu^L$')
    )
    w_df['IRL Method'] = (
        w_df['IRL Method'].str.replace('wL', r'$w^L$')
    )
    w_hue_order = (
        pd.Series(w_hue_order).str.replace('wL', r'$w^L$')
    )

    mu_df = mu_df.sort_values(['Demo Producer', 'Feature Expectation'])
    w_df = w_df.sort_values(['Weight', 'IRL Method'])

    # Plot boxplot for feature expectations
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(5, 8.5),
        height_ratios=[2, 1],
    )
    sns.boxplot(
        x=mu_df['Feature Expectation'],
        y=mu_df['Value'],
        hue=mu_df['Demo Producer'],
        # hue_order=mu_hue_order,
        ax=ax1,
        fliersize=0,  # Remove outliers
        saturation=1,
        palette=list([cp[0]]) + list(cp[3:6]) + list(cp[7:]),
        boxprops=dict(alpha=1),
        linewidth=1,
    )
    ax1.set_ylabel(None)
    ax1.set_xlabel('Learned Feature Expectations')
    ax1.set_ylim([-.1,1.1])
    ax1.get_legend().remove()
    ax1.legend(
        title=None,
        ncol=2,
        # fontsize=12,
        loc='lower left',
        # bbox_to_anchor=(1.0, -.25),
    )

    # Plot boxplot for weights
    sns.boxplot(
        x=w_df['Weight'],
        y=w_df['Value'],
        hue=w_df['IRL Method'],
        # hue_order=w_hue_order,
        ax=ax2,
        fliersize=0,  # Remove outliers
        saturation=1,
        boxprops=dict(alpha=1),
        linewidth=1,
        palette=list(cp[1:3]) + list(cp[6:]),
    )
    ax2.set_ylabel(None)
    ax2.set_xlabel('Learned Weights')
    ax2.set_ylim([-1, 1])
    ax2.get_legend().remove()
    ax2.legend(
        title=None,
        ncol=2,
        # fontsize=19.5,
        loc='lower left',
        # bbox_to_anchor=(.85, -.35),
     )

    ax1.set_title(f"Expert Algo: {expert_algo}\n{28*'-'}")
    plt.tight_layout()

    print(f"SOURCE DATASET: {source_dataset}")
    print(f"TARGET DATASET: {target_dataset}")

    return mu_df, w_df
