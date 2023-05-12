import itertools
import logging
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.base import BaseEstimator, ClassifierMixin

class OptimizationProblem():

    def __init__(self, name, c, A_eq=None, b_eq=None, b_ub=None, A_ub=None):
        self.name = name
        self.c = c
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.b_ub = b_ub
        self.A_ub = A_ub

class ObjectiveSplit():

    def __init__(self, name, parent, c, b_ub=None, A_ub=None):
        self.name = name
        self.parent = parent
        self.c = c
        self.b_ub = b_ub
        self.A_ub = A_ub

class Objective():

    def __init__(self):
        pass

    def fit(self, ldf):
        raise NotImplementedError()

    def to_splits(self):
        raise NotImplementedError()


class LinearObjective(Objective):

    def __init__(self):
        self.n_splits = 1
        super().__init__()

    def fit(self, ldf):
        self.b_ub_row_ = None
        self.c_ = self._construct_reward(ldf)
        return self

    def to_splits(self):
        """
        Returns objective as one or more splits.

        Parameters
        ----------
        None

        Returns
        -------
        array<ObjectiveSplit>
        """
        split = ObjectiveSplit(
            name=self.name,
            parent=self,
            c=self.c_,
            A_ub=None,
            b_ub=None,
        )
        return [split]

    def _construct_reward(self):
        raise NotImplementedError


class AbsoluteValueObjective(Objective):

    def __init__(self):
        self.n_splits = 2
        super().__init__()

    def fit(self, ldf):
        self.b_ub_row_ = 0  # Numb of constraints
        self.A_ub_row__split1_ = self._compute_A_ub_row__split1(ldf)
        self.A_ub_row__split2_ = self._compute_A_ub_row__split2(ldf)
        self.c__split1_ = self._construct_reward__split1(ldf)
        self.c__split2_ = self._construct_reward__split2(ldf)
        return self

    def to_splits(self):
        """
        Returns objective as one or more splits.

        Parameters
        ----------
        None

        Returns
        -------
        array<ObjectiveSplit>
        """
        split1 = ObjectiveSplit(
            name=f"{self.name} Split1",
            parent=self,
            c=self.c__split1_,
            b_ub=self.b_ub_row_,
            A_ub=self.A_ub_row__split1_,
        )
        split2 = ObjectiveSplit(
            name=f"{self.name} Split2",
            parent=self,
            c=self.c__split2_,
            b_ub=self.b_ub_row_,
            A_ub=self.A_ub_row__split2_,
        )
        return [split1, split2]

    def _compute_A_ub_row__split1(self, ldf):
        raise NotImplementedError()

    def _compute_A_ub_row__split2(self, ldf):
        raise NotImplementedError()

    def _construct_reward__split1(self, ldf):
        raise NotImplementedError()

    def _construct_reward__split2(self, ldf):
        raise NotImplementedError()


class AccuracyObjective(LinearObjective):

    def __init__(self):
        self.name = 'Acc'
        super().__init__()

    def _construct_reward(self, ldf):
        """
        Constructs the reward function when the objective is accuracy.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        ------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        ldf['r'] = (ldf['yhat'] == ldf['y']).astype(float)
        c = -1 * ldf['r']  # Negative since maximizing not minimizing
        return c


class DisparateImpactObjective(AbsoluteValueObjective):

    def __init__(self):
        self.name = 'DispImp'
        super().__init__()

    def _compute_A_ub_row__split1(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=1|z=0) >= P(yhat=1|z=1)
            ```
        which is Disparate Impact opt_problem 1.

        Parameters
        ----------
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        n_actions = 2
        ldf = ldf.copy()
        filt__yhat1_giv_z0 = (ldf['z'] == 0) & (ldf['yhat'] == 1)
        filt__yhat1_giv_z1 = (ldf['z'] == 1) & (ldf['yhat'] == 1)
        ldf['A_ub'] = 0.0
        ldf.loc[filt__yhat1_giv_z0, 'A_ub'] = -1
        ldf.loc[filt__yhat1_giv_z1, 'A_ub'] = 1
        return ldf['A_ub']

    def _compute_A_ub_row__split2(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=1|z=1) >= P(yhat=1|z=0)
            ```
        which is Disparate Impact opt_problem 2.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.


        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        n_actions = 2
        ldf = ldf.copy()
        filt__yhat1_giv_z0 = (ldf['z'] == 0) & (ldf['yhat'] == 1)
        filt__yhat1_giv_z1 = (ldf['z'] == 1) & (ldf['yhat'] == 1)
        ldf['A_ub'] = 0.0
        ldf.loc[filt__yhat1_giv_z0, 'A_ub'] = 1
        ldf.loc[filt__yhat1_giv_z1, 'A_ub'] = -1
        return ldf['A_ub']

    def _construct_reward__split1(self, ldf):
        """
        Constructs the reward function for Disparate Impact opt_problem 1.
        opt_problem 1 is when we constrain P(yhat=1|z=0) >= P(yhat=1|z=1), in
        which case the reward penalizes giving the Z=0 group the positive
        prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_giv_z0 = (ldf['z'] == 0) & (ldf['yhat'] == 1)
        filt__yhat1_giv_z1 = (ldf['z'] == 1) & (ldf['yhat'] == 1)
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_giv_z0, 'r'] = -2
        ldf.loc[filt__yhat1_giv_z1, 'r'] = 2
        c = -1 * ldf['r']  # Negative since maximizing not minimizing
        return c

    def _construct_reward__split2(self, ldf):
        """
        Constructs the reward function for Disparate Impact opt_problem 2.
        opt_problem 2 is when we constrain P(yhat=1|z=1) >= P(yhat=1|z=0), in
        which case the reward penalizes giving the Z=1 group the positive
        prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_giv_z0 = (ldf['z'] == 0) & (ldf['yhat'] == 1)
        filt__yhat1_giv_z1 = (ldf['z'] == 1) & (ldf['yhat'] == 1)
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_giv_z0, 'r'] = 2
        ldf.loc[filt__yhat1_giv_z1, 'r'] = -2
        c = -1 * ldf['r']  # Negative since maximizing not minimizing
        return c


class EqualOpportunityObjective(AbsoluteValueObjective):

    def __init__(self):
        self.name = 'EqOpp'
        super().__init__()

    def _compute_A_ub_row__split1(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=1|y=1,z=0) >= P(yhat=1|y=1,z=1)
            ```
        which is Equal Opportunity opt_problem 1.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        n_actions = 2
        ldf = ldf.copy()
        filt__yhat1_giv_y1_z0 = (ldf['z'] == 0) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        filt__yhat1_giv_y1_z1 = (ldf['z'] == 1) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        ldf['A_ub'] = 0.0
        ldf.loc[filt__yhat1_giv_y1_z0, 'A_ub'] = -1
        ldf.loc[filt__yhat1_giv_y1_z1, 'A_ub'] = 1
        return ldf['A_ub']

    def _compute_A_ub_row__split2(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=1|y=1,z=1) >= P(yhat=1|y=1,z=0)
            ```
        which is Equal Opportunity opt_problem 2.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        n_actions = 2
        ldf = ldf.copy()
        filt__yhat1_giv_y1_z0 = (ldf['z'] == 0) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        filt__yhat1_giv_y1_z1 = (ldf['z'] == 1) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        ldf['A_ub'] = 0.0
        ldf.loc[filt__yhat1_giv_y1_z0, 'A_ub'] = 1
        ldf.loc[filt__yhat1_giv_y1_z1, 'A_ub'] = -1
        return ldf['A_ub']

    def _construct_reward__split1(self, ldf):
        """
        Constructs the reward function for Equal Opportunity  opt_problem 1.
        opt_problem 1 is when we constrain P(yhat=1|y=1,z=0) >= P(yhat=1|y=1,z=1), in
        which case the reward penalizes giving the Z=0 group the positive
        prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_giv_y1_z0 = (ldf['z'] == 0) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        filt__yhat1_giv_y1_z1 = (ldf['z'] == 1) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_giv_y1_z0, 'r'] = -5
        ldf.loc[filt__yhat1_giv_y1_z1, 'r'] = 5
        c = -1 * ldf['r']  # Negative since maximizing not minimizing
        return c

    def _construct_reward__split2(self, ldf):
        """
        Constructs the reward function for Disparate Impact opt_problem 1.
        opt_problem 1 is when we constrain P(yhat=1|y=1,z=0) >= P(yhat=1|y=1,z=1), in
        which case the reward penalizes giving the Z=0 group the positive
        prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_giv_y1_z0 = (ldf['z'] == 0) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        filt__yhat1_giv_y1_z1 = (ldf['z'] == 1) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_giv_y1_z0, 'r'] = 5
        ldf.loc[filt__yhat1_giv_y1_z1, 'r'] = -5
        c = -1 * ldf['r']  # Negative since maximizing not minimizing
        return c


class ObjectiveSet():

    def __init__(self, objectives):
        self.objectives = objectives

    def construct_opt_problems(self, reward_weights, ldf, A_eq, b_eq):
        self.objectives = [obj.fit(ldf) for obj in self.objectives]

        abs_val_splits = []
        linear_splits = []
        abs_val_split_generator = []
        counter = 0
        for obj in self.objectives:
            if obj.n_splits == 1:
                linear_splits += obj.to_splits()
            if obj.n_splits == 2:
                abs_val_splits += obj.to_splits()
                abs_val_split_generator.append([counter, counter+1])
                counter += 2


        # abs_val_splits_perms = list(itertools.product(*[[0,1], [2,3]]))
        # >>> [(0, 2), (0, 3), (1, 2), (1, 3)]
        abs_val_splits_perms = list(itertools.product(*abs_val_split_generator))
        opt_problems = []
        for split_indexes in abs_val_splits_perms:
            _abs_val_splits = [abs_val_splits[idx] for idx in split_indexes]
            all_splits = linear_splits + _abs_val_splits
            name = ','.join([split.name for split in all_splits])
            A_ub = np.array([split.A_ub for split in _abs_val_splits], dtype=float)
            b_ub = np.array([split.b_ub for split in _abs_val_splits], dtype=float)
            c = reward_weights[all_splits[0].parent.name] * all_splits[0].c
            for split in all_splits[1:]:
                c += reward_weights[split.parent.name] * split.c

            opt_problem = OptimizationProblem(
                name=name,
                A_eq=A_eq,
                b_eq=b_eq,
                A_ub=A_ub,
                b_ub=b_ub,
                c=c,
            )

            opt_problems.append(opt_problem)

        return opt_problems


class ClassificationMDP:
    """
    Parameters
    ----------
    gamma : float, range, [0, 1)
        The discount factor. Should be close to zero since this is a
        classification mdp, but cannot be zero exactly otherwise the linear
        program won't converge.
    x_cols : list<str>
        The names of the classification features (excluding 'z').
    acc_reward_weight : float
        The Accuracy reward weight.
    disp_imp_reward_weight : float
        The Disparate Impact reward weight.
    eq_opp__reward_weight : float
        The Equal Opportunity reward weight.

    Attributes
    ----------
    b_eq_ : numpy.array<float>
        A.k.a. "mu0". Initial state probabilities.
    A_eq_ : numpy.ndarray<float>, shape (len(df), 2*len(df))
        The state-action transition matrix.
    n_states_ : int
        Number of states.
    state_df_ : pandas.DataFrame
        Index is state index, columns are features, mu0, and optimal policy
        actions.
    state_lookup_ : dict<tuple, int>
        Maps classification features to its MDP state index.
    ldf_ : pandas.DataFrame
        "Lambda dataframe". One row for each state and action combination.
        Columns are **x_cols, z, y, yhat.
    b_ub_disp_imp_ : np.array<float>, len(n_states_)
        The uppber bound `b` for Disparate Impact Split1 and Split2.
    opt_problems_ : array<dict>
        Array of each linear optimization "sub" problem to solve. Each
        opt_problem is a dict with the following structure:
            'A_eq': self.A_eq_,
            'b_eq': self.b_eq_,
            'A_ub': numpy.ndarray<float>, len(num feat exp with abs val)
                The upper bound linear equation constraints for the split that
                occurs as the result of absolute value signs in the feat exp.
            'b_ub': np.array<float>, len(num feat exp with abs val)
                The uppber bound `b` for the linear equation constraint for the
                split that occurs as a result of absolute value signs in the
                feat exp.
            'c': array-like<float>, len(n_states*n_actions)
                Reward vector for the opt_problem.
    """

    def __init__(
        self, gamma, x_cols, acc_reward_weight, disp_imp_reward_weight, eq_opp_reward_weight):
        self.gamma = gamma
        self.x_cols = x_cols
        self.acc_reward_weight = acc_reward_weight
        self.disp_imp_reward_weight = disp_imp_reward_weight
        self.eq_opp_reward_weight = eq_opp_reward_weight
        self.state_df_ = None
        self.state_lookup_ = None
        self.n_states = None
        self.A_eq_ = None
        self.b_eq_ = None
        self.ldf_ = None
        self.opt_problems_ = None

    def fit(self, clf_df):
        """
        Sets Attributes
        ---------------
        b_eq_
        A_eq_
        state_lookup_
        n_states_
        ldf_
        opt_problems_

        Returns
        -------
        None
        """
        clf_df = clf_df.copy()

        # Generate the state_df_ and the X,y -> state mapping
        self.state_df_ = (
            clf_df
            .groupby(self.x_cols+['z', 'y']).size()
            .reset_index().rename(columns={0: 'count'})
        )
        self.state_lookup_ = {}
        state_counter = 0
        for idx, row in self.state_df_.iloc[:, :-1].iterrows():
            self.state_lookup_[tuple(row)] = state_counter
            state_counter += 1

        # Cache n_states since frequently used in computations
        self.n_states_ = state_counter

        # Compute `b_eq` === `mu0` (initial state probabilities)
        self.b_eq_ = self.state_df_['count'] / self.state_df_['count'].sum()
        self.state_df_['mu0'] = self.b_eq_
        self.state_df_ = self.state_df_.drop(columns='count')

        # Compute transition matrix linear equations `A_eq`
        self.A_eq_ = self._compute_A_eq(self.b_eq_)

        # Generate the lambda dataframe (state-action indexed)
        self.ldf_ = self._compute_lambda_linear_equations()

        accur = AccuracyObjective().fit(self.ldf_)
        dis_imp = DisparateImpactObjective().fit(self.ldf_)
        equ_opp = EqualOpportunityObjective().fit(self.ldf_)

        reward_weights = {
            'Acc': self.acc_reward_weight,
            'DispImp': self.disp_imp_reward_weight,
            'EqOpp': self.eq_opp_reward_weight,
        }
        objset = ObjectiveSet([accur, dis_imp, equ_opp])
        self.opt_problems_ = objset.construct_opt_problems(
            reward_weights=reward_weights,
            ldf=self.ldf_,
            A_eq=self.A_eq_,
            b_eq=self.b_eq_,
        )

        # ## Combine all constraints for each opt_problem permutation
        # self.opt_problems_ = [
        #     {
        #         'names': [dis_imp.name+'1', equ_opp.name+'1'],
        #         'A_eq': self.A_eq_,
        #         'b_eq': self.b_eq_,
        #         'A_ub': np.array([
        #             dis_imp.A_ub_row__split1_,
        #             equ_opp.A_ub_row__split1_,
        #         ], dtype=float),
        #         'b_ub': np.array([
        #             dis_imp.b_ub_row_,
        #             equ_opp.b_ub_row_,
        #         ], dtype=float),
        #         'c': (
        #             self.acc_reward_weight * accur.c_
        #             + self.disp_imp_reward_weight * dis_imp.c__split1_
        #             + self.eq_opp_reward_weight * equ_opp.c__split1_
        #         ),
        #     }, {
        #         'names': [dis_imp.name+'1', equ_opp.name+'2'],
        #         'A_eq': self.A_eq_,
        #         'b_eq': self.b_eq_,
        #         'A_ub': np.array([
        #             dis_imp.A_ub_row__split1_,
        #             equ_opp.A_ub_row__split2_,
        #         ], dtype=float),
        #         'b_ub': np.array([
        #             dis_imp.b_ub_row_,
        #             equ_opp.b_ub_row_,
        #         ], dtype=float),
        #         'c': (
        #             self.acc_reward_weight * accur.c_
        #             + self.disp_imp_reward_weight * dis_imp.c__split1_
        #             + self.eq_opp_reward_weight * equ_opp.c__split2_
        #         ),
        #     }, {
        #         'names': [dis_imp.name+'2', equ_opp.name+'1'],
        #         'A_eq': self.A_eq_,
        #         'b_eq': self.b_eq_,
        #         'A_ub': np.array([
        #             dis_imp.A_ub_row__split2_,
        #             equ_opp.A_ub_row__split1_,
        #         ], dtype=float),
        #         'b_ub': np.array([
        #             dis_imp.b_ub_row_,
        #             equ_opp.b_ub_row_,
        #         ], dtype=float),
        #         'c': (
        #             self.acc_reward_weight * accur.c_
        #             + self.disp_imp_reward_weight * dis_imp.c__split2_
        #             + self.eq_opp_reward_weight * equ_opp.c__split1_
        #         ),
        #     }, {
        #         'names': [dis_imp.name+'2', equ_opp.name+'2'],
        #         'A_eq': self.A_eq_,
        #         'b_eq': self.b_eq_,
        #         'A_ub': np.array([
        #             dis_imp.A_ub_row__split2_,
        #             equ_opp.A_ub_row__split2_,
        #         ], dtype=float),
        #         'b_ub': np.array([
        #             dis_imp.b_ub_row_,
        #             equ_opp.b_ub_row_,
        #         ], dtype=float),
        #         'c': (
        #             self.acc_reward_weight * accur.c_
        #             + self.disp_imp_reward_weight * dis_imp.c__split2_
        #             + self.eq_opp_reward_weight * equ_opp.c__split2_
        #         ),
        #     },
        # ]

        return None

    def compute_optimal_policies(self, skip_error_terms=False, method='highs'):
        """
        Computes the optimal policies for the classification MDP.

        Parameters
        ----------
        skip_error_terms : bool, default False
            If true, doesn't try and find all solutions and instead just invokes
            the scipy solver on the input terms.
        method : str, default 'highs'
            The scipy solver method to use. Options are 'highs' (default),
            'highs-ds', 'highs-ipm'.

        Returns
        -------
        opt_pols : list<np.array>
            Optimal policies.
        """
        # Find the best policy/reward of all the opt_problems
        best_policies_best_rewards = []
        for subprob in self.opt_problems_:
            # opt_pols, opt_rew = _find_all_solutions_lp(
            #     c=subprob['c'],
            #     A_eq=subprob['A_eq'],
            #     b_eq=subprob['b_eq'],
            #     A_ub=subprob['A_ub'],
            #     b_ub=subprob['b_ub'],
            #     error_term=1e-12,
            #     skip_error_terms=skip_error_terms,
            #     method=method,
            # )
            opt_pols, opt_rew = _find_all_solutions_lp(
                c=subprob.c,
                A_eq=subprob.A_eq,
                b_eq=subprob.b_eq,
                A_ub=subprob.A_ub,
                b_ub=subprob.b_ub,
                error_term=1e-12,
                skip_error_terms=skip_error_terms,
                method=method,
            )
            best_policies_best_rewards.append({
                'policies': list(opt_pols),
                'reward': np.round(opt_rew, decimals=6),
            })

        opt_pols, opt_rew = _find_best_policies_from_multiple_opt_problems(
            best_policies_best_rewards,
        )

        # Append optimal policy actions to state_df_ attribute
        for i, pi in enumerate(opt_pols):
            self.state_df_[f"pi_{i}"] = pi

        if logging.DEBUG >= logging.root.level:
            logging.debug(f"Best Reward: {opt_rew:.3f}")
            logging.debug(f"{len(opt_pols)} optimal policies found")

            state_df = (
                self.ldf_.copy()[self.x_cols + ['z', 'y']].drop_duplicates()
            )

            # display(state_df.head(5))

            for pi in state_df.columns[2:5]:
                logging.debug(pi)
                display(state_df.groupby(['z', 'y'])[[pi]].agg(['count', 'sum']))
                # Print accuracy
                acc = _compute_accuracy(state_df, pi)
                logging.debug(f"Accuracy: {acc:.3f}")
                filt__yhat1_giv_z0 = (state_df['z'] == 0) & (state_df[pi] == 1)
                filt__yhat1_giv_z1 = (state_df['z'] == 1) & (state_df[pi] == 1)
                filt__z0 = state_df['z'] == 0
                filt__z1 = state_df['z'] == 1
                p_yhat1_giv_z0 = len(state_df[filt__yhat1_giv_z0])/len(state_df[filt__z0])
                p_yhat1_giv_z1 = len(state_df[filt__yhat1_giv_z1])/len(state_df[filt__z1])
                dispimp = np.abs(p_yhat1_giv_z0 - p_yhat1_giv_z1)
                # Print Disparate Impact
                logging.debug(f"Disparate Impact: {dispimp:.3f}")

        return opt_pols

    def _compute_A_eq(self, b_eq):
        """
        Constructs a set of linear equations representing the state-action
        transition probabilities  where all "states" (classification dataset
        samples) have the initial state probability, regardless of the action
        taken.
        "Classification MDP".

        Returns
        -------
        A_eq_
        """
        n_states = len(b_eq)
        n_actions = 2
        A_eq = np.zeros((n_states, n_states*n_actions))
        for s in range(n_states):
            for sp in range(n_states):
                for a in range(n_actions):
                    if s == sp:
                        A_eq[s][sp*n_actions+a] = 1 - self.gamma*b_eq[sp]
                    else:
                        A_eq[s][sp*n_actions+a] = 0 - self.gamma*b_eq[sp]
        return A_eq

    def _compute_lambda_linear_equations(self):
        """
        TODO
        """
        state_df = self.state_df_.copy()
        ldf = pd.concat([state_df, state_df], axis=0).reset_index(drop=True)
        ldf = ldf.sort_values(list(ldf.columns))  # Set every two rows the same
        yhat = np.zeros(len(ldf), dtype=int)
        yhat[1::2] = 1  # Makes 'a' 0, 1 repeating sequence
        ldf['yhat'] = yhat
        return ldf


class ClassificationMDPPolicy(BaseEstimator, ClassifierMixin):
    """
    A Scikit-Learn compatible wrapper for a Classification MDP policy.

    Parameters
    ----------
    mdp : ClassificationMDP
        The classification mdp instance.
    pi : numpy.array<int>
        The optimal policy (optimal action once `y` is known).
    clf : sklearn.BaseEstimator
        Binary classifier used for predicting `y` from `X`. It needs to be
        fitted already, and it needs to contain the full preprocessing of
        inputs.
    default_action : int
        The default action to use if the state lookup fails.

    Attributes
    ----------
    reward_weights : list<float>
        Weights for each objective component.
    """

    def __init__(self, mdp, pi, clf, default_action=0):
        self.mdp = mdp
        self.pi = pi
        self.clf = clf
        self.default_action = default_action

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

    def predict(self, X, y=None):
        """
        If `y` is None, then predicts `y` from `X`, then returns the optimal action for that value
        of `(X, y)`. Sort of like a two-step POMDP.

        If `y` is provided, then does not predict `y`.

        I.e. here is some crude pseudocode reprsenting what's actually
        happening:
            ```
            if y is None:
                y = predict(X)

            a = pi(X, y)
            return a
            ```
        Parameters
        ---------
        X : pandas.DataFrame
            Input data.
        y : pandas.Series
            label data.

        Returns
        -------
        actions : numpy.array<int>, len(len(X))
            The "predictions", actually the actions from the Clf MDP.
        """
        df = pd.DataFrame(X)
        # By using `predict_proba` and inserting randomness, we ensure that the
        # assumed y values are not always the majority.
        # df['y'] = (
        #     self.clf.predict_proba(X)[:,0] >= np.random.rand(len(X))
        # ).astype(int)
        if y is None:
            df['y'] = self.clf.predict(X)
        else:
            df['y'] = y

        actions = np.zeros(len(X))

        # Get rid of any unused columns otherwise the state lookup breaks.
        df = df[self.mdp.x_cols + ['z', 'y']]

        # if logging.DEBUG >= logging.root.level:
        #     display(df)
        #     display(self.mdp.state_lookup_)

        for i, (idx, row) in enumerate(df.iterrows()):
            try:
                state = self.mdp.state_lookup_[tuple(row)]
                actions[i] = self.pi[state]
            except KeyError as e:
                logging.info('\tState Lookup Error: ' + str(e))
                logging.info(f"\tUsing default action: {self.default_action}")
                actions[i] = self.default_action

        return actions


def _find_best_policies_from_multiple_opt_problems(best_policies_best_rewards):
    """
    Parameters
    ----------
    best_policies_best_rewards : tuple<dict>
        Example:
        ```
        best_policies_best_rewards =
            {
                'policies': list(opt_pols_disp_imp_split1_adult),
                'reward': np.round(opt_rew_disp_imp_split1_adult, decimals=6)
            },
            {
                'policies': list(opt_pols_disp_imp_split2_adult),
                'reward': np.round(opt_rew_disp_imp_split2_adult, decimals=6)
            },
        )
        ```

    Returns
    -------
    best_of_best_pols : list<numpy.array>
        The unique list of optimal policies from all opt_problems.
    best_of_best_reward : float
        The best reward of all opt_problems.
    """
    rewards = [bpbr['reward'] for bpbr in best_policies_best_rewards]
    best_idx = np.argwhere(rewards == np.amax(rewards)).flatten().tolist()
    logging.debug(f"best_idx: {best_idx}")
    best_of_best_pols = []
    # For each opt_problem index where the reward is the best reward
    for idx in best_idx:
        # Get all the policies from that opt_problem (all have same reward)
        pols = best_policies_best_rewards[idx]['policies']
        # For each of these policies, check add it to the list of
        # best_of_best_pols if it's not already in it.
        for pol in pols:
            logging.debug(f"pol {pol}")
            pol_in_best = False
            for bpol in best_of_best_pols:
                logging.debug("\tbpol {bpol}")
                if np.allclose(pol, bpol, atol=1e-5):
                    logging.debug("\t\t" + f"{pol} already in best_of_best_pols")
                    pol_in_best = True
                    break
            if not pol_in_best:
                logging.debug(f"appending {pol} to best_of_best_pols")
                best_of_best_pols.append(pol)
    best_of_best_reward = rewards[best_idx[0]]
    return best_of_best_pols, best_of_best_reward


def _find_all_solutions_lp(
        c, A_eq, b_eq, A_ub=None, b_ub=None, error_term=1e-12, skip_error_terms=False, method='highs'):
    """
    Wrapper around scipy.optimize.linprog that finds ALL optimal solutions
    by iteratively solving the LP problem after adding/subtracting an "error"
    term to each objective component.

    Parameters
    ----------
    c : 1-D array
        The coefficients of the linear objective function to be minimized.
    A_eq : 2-D array
        The equality constraint matrix. Each row of ``A_eq`` specifies the
        coefficients of a linear equality constraint on ``x``.
    b_eq : 1-D array
        The equality constraint vector. Each element of ``A_eq @ x`` must equal
        the corresponding element of ``b_eq``.
    A_ub : 2-D array, optional
        The inequality constraint matrix. Each row of ``A_ub`` specifies the
        coefficients of a linear inequality constraint on ``x``.
    b_ub : 1-D array, optional
        The inequality constraint vector. Each element represents an
        upper bound on the corresponding value of ``A_ub @ x``.
    error_term : float, default 1e-12
        Allowed error from the optimal reward to still be considered optimal.
    skip_error_terms : bool, default False
        If true, doesn't try and find all solutions and instead just invokes
        the scipy solver on the input terms.
    method : str, default 'highs'
        The scipy solver method to use. Options are 'highs' (default),
        'highs-ds', 'highs-ipm'.

    Returns
    -------
    best_policies : list<numpy.array>
        List of the policies that have the optimal reward.
    best_reward : float
        The optimal reward.
    """
    best_policies = []
    best_reward = -1*np.inf
    n_states = len(b_eq)
    n_actions = 2

    if skip_error_terms:
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub)
        best_reward = -1*res.fun
        pi_opt = np.zeros(n_states, dtype=int)
        for s in range(n_states):
            start_idx = s*n_actions
            end_idx = s*n_actions+n_actions
            pi_opt[s] = res.x[start_idx:end_idx].argmax()

        best_policies = [pi_opt]

    else:
        for i in range(len(A_eq)):

            # Positive error
            cpos = np.array(c)
            cpos[i] += error_term
            res = linprog(cpos, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub)
            if ((-1*res.fun > best_reward)
                and (not np.isclose(-1*res.fun, best_reward, atol=.001))):
                best_reward = -1*res.fun
                logging.debug(f"\nBest Reward:\t {best_reward}")
                logging.debug(f"Lambdas:\t {np.round(res.x, 2)}")
            pi_opt = np.zeros(n_states, dtype=int)
            for s in range(n_states):
                start_idx = s*n_actions
                end_idx = s*n_actions+n_actions
                pi_opt[s] = res.x[start_idx:end_idx].argmax()
            if not _is_pol_in_pols(pi_opt, best_policies):
                best_policies.append(pi_opt)
                logging.debug(f"Optimal Policy:\t, {pi_opt} \n")

            # Negative error
            cneg = np.array(c)
            cneg[i] -= error_term
            res = linprog(cneg, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub)
            if ((-1*res.fun > best_reward)
                and (not np.isclose(-1*res.fun, best_reward, atol=.001))):
                best_reward = -1*res.fun
                logging.debug(f"\nBest Reward:\t {best_reward}")
            logging.debug(f"Lambdas:\t {np.round(res.x, 2)}")
            pi_opt = np.zeros(n_states, dtype=int)
            for s in range(n_states):
                start_idx = s*n_actions
                end_idx = s*n_actions+n_actions
                pi_opt[s] = res.x[start_idx:end_idx].argmax()
            if not _is_pol_in_pols(pi_opt, best_policies):
                best_policies.append(pi_opt)
                logging.debug(f"Optimal Policy:\t, {pi_opt} \n")

        logging.debug('\nOptimal policies:')
        for pi in best_policies:
            logging.debug(f"\t{np.round(pi, 2)}")

    return best_policies, best_reward

def _is_pol_in_pols(pol, policies):
    """
    Check if a policy is in a list of policies.

    Parameters
    ----------
    pol : array-like
        Candidate policy.
    policies : 2d array-like
        List of policies to check if candidate policy is in.

    Returns
    -------
    bool
        Whether candidate policy is in list of policies.

    """
    for p in policies:
        if np.array_equal(p, pol):
            return True
    return False


def _compute_accuracy(df, yhat_col):
    acc = (df['y'] == df[yhat_col]).mean()
    return acc
