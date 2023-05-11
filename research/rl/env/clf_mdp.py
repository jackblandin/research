import logging
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.base import BaseEstimator, ClassifierMixin


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
        The uppber bound `b` for Disparate Impact Sub1 and Sub2.
    subproblems_ : array<dict>
        Array of each linear optimization "sub" problem to solve. Each
        subproblem is a dict with the following structure:
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
                Reward vector for the subproblem.
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
        self.subproblems_ = None

    def fit(self, clf_df):
        """
        Sets Attributes
        ---------------
        b_eq_
        A_eq_
        state_lookup_
        n_states_
        ldf_
        subproblems_

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

        # Compute the extra constraints for the two Disparate Impact subproblems
        b_ub_row__disp_imp = 0  # Numb of constraints
        A_ub_row__disp_imp_sub1 = self._compute_A_ub_row__disp_imp_sub1()
        A_ub_row__disp_imp_sub2 = self._compute_A_ub_row__disp_imp_sub2()

        # Compute the extra constraints for the two Disparate Impact subproblems
        b_ub_row__eq_opp = 0  # Numb of constraints
        A_ub_row__eq_opp_sub1 = self._compute_A_ub_row__eq_opp_sub1()
        A_ub_row__eq_opp_sub2 = self._compute_A_ub_row__eq_opp_sub2()

        ## Combine all constraints for each subproblem permutation
        self.subproblems_ = [
            {
                'name': 'DispImp1, EqOpp1',
                'A_eq': self.A_eq_,
                'b_eq': self.b_eq_,
                'A_ub': np.array([A_ub_row__disp_imp_sub1, A_ub_row__eq_opp_sub1], dtype=float),
                'b_ub': np.array([b_ub_row__disp_imp, b_ub_row__eq_opp], dtype=float),
                'c': (
                    self.acc_reward_weight * self._construct_reward__accuracy()
                    + self.disp_imp_reward_weight * self._construct_reward__disp_imp_sub1()
                    + self.eq_opp_reward_weight * self._construct_reward__eq_opp_sub1()
                ),
            }, {
                'name': 'DispImp1, EqOpp2',
                'A_eq': self.A_eq_,
                'b_eq': self.b_eq_,
                'A_ub': np.array([A_ub_row__disp_imp_sub1, A_ub_row__eq_opp_sub2], dtype=float),
                'b_ub': np.array([b_ub_row__disp_imp, b_ub_row__eq_opp], dtype=float),
                'c': (
                    self.acc_reward_weight * self._construct_reward__accuracy()
                    + self.disp_imp_reward_weight * self._construct_reward__disp_imp_sub1()
                    + self.eq_opp_reward_weight * self._construct_reward__eq_opp_sub2()
                ),
            }, {
                'name': 'DispImp2, EqOpp1',
                'A_eq': self.A_eq_,
                'b_eq': self.b_eq_,
                'A_ub': np.array([A_ub_row__disp_imp_sub2, A_ub_row__eq_opp_sub1], dtype=float),
                'b_ub': np.array([b_ub_row__disp_imp, b_ub_row__eq_opp], dtype=float),
                'c': (
                    self.acc_reward_weight * self._construct_reward__accuracy()
                    + self.disp_imp_reward_weight * self._construct_reward__disp_imp_sub2()
                    + self.eq_opp_reward_weight * self._construct_reward__eq_opp_sub1()
                ),
            }, {
                'name': 'DispImp2, EqOpp2',
                'A_eq': self.A_eq_,
                'b_eq': self.b_eq_,
                'A_ub': np.array([A_ub_row__disp_imp_sub2, A_ub_row__eq_opp_sub2], dtype=float),
                'b_ub': np.array([b_ub_row__disp_imp, b_ub_row__eq_opp], dtype=float),
                'c': (
                    self.acc_reward_weight * self._construct_reward__accuracy()
                    + self.disp_imp_reward_weight * self._construct_reward__disp_imp_sub2()
                    + self.eq_opp_reward_weight * self._construct_reward__eq_opp_sub2()
                ),
            },
        ]

        return None

    def compute_optimal_policies(self):
        """
        Computes the optimal policies for the classification MDP.

        Parameters
        ----------

        Returns
        -------
        opt_pols : list<np.array>
            Optimal policies.
        """
        # Find the best policy/reward of all the subproblems
        best_policies_best_rewards = []
        for subprob in self.subproblems_:
            opt_pols, opt_rew = _find_all_solutions_lp(
                c=subprob['c'],
                A_eq=subprob['A_eq'],
                b_eq=subprob['b_eq'],
                A_ub=subprob['A_ub'],
                b_ub=subprob['b_ub'],
                error_term=1e-12,
            )
            best_policies_best_rewards.append({
                'policies': list(opt_pols),
                'reward': np.round(opt_rew, decimals=6),
            })

        opt_pols, opt_rew = _find_best_policies_from_multiple_subproblems(
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

    def _compute_A_ub_row__disp_imp_sub1(self):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=1|z=0) >= P(yhat=1|z=1)
            ```
        which is Disparate Impact Subproblem 1.

        Parameters
        ----------

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        n_actions = 2
        ldf = self.ldf_.copy()
        filt__yhat1_giv_z0 = (ldf['z'] == 0) & (ldf['yhat'] == 1)
        filt__yhat1_giv_z1 = (ldf['z'] == 1) & (ldf['yhat'] == 1)
        ldf['A_ub'] = 0.0
        ldf.loc[filt__yhat1_giv_z0, 'A_ub'] = -1
        ldf.loc[filt__yhat1_giv_z1, 'A_ub'] = 1
        return ldf['A_ub']

    def _compute_A_ub_row__disp_imp_sub2(self):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=1|z=1) >= P(yhat=1|z=0)
            ```
        which is Disparate Impact Subproblem 2.

        Parameters
        ----------

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        n_actions = 2
        ldf = self.ldf_.copy()
        filt__yhat1_giv_z0 = (ldf['z'] == 0) & (ldf['yhat'] == 1)
        filt__yhat1_giv_z1 = (ldf['z'] == 1) & (ldf['yhat'] == 1)
        ldf['A_ub'] = 0.0
        ldf.loc[filt__yhat1_giv_z0, 'A_ub'] = 1
        ldf.loc[filt__yhat1_giv_z1, 'A_ub'] = -1
        return ldf['A_ub']

    def _compute_A_ub_row__eq_opp_sub1(self):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=1|y=1,z=0) >= P(yhat=1|y=1,z=1)
            ```
        which is Equal Opportunity subproblem 1.

        Parameters
        ----------

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        n_actions = 2
        ldf = self.ldf_.copy()
        filt__yhat1_giv_y1_z0 = (ldf['z'] == 0) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        filt__yhat1_giv_y1_z1 = (ldf['z'] == 1) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        ldf['A_ub'] = 0.0
        ldf.loc[filt__yhat1_giv_y1_z0, 'A_ub'] = -1
        ldf.loc[filt__yhat1_giv_y1_z1, 'A_ub'] = 1
        return ldf['A_ub']

    def _compute_A_ub_row__eq_opp_sub2(self):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=1|y=1,z=1) >= P(yhat=1|y=1,z=0)
            ```
        which is Equal Opportunity subproblem 2.

        Parameters
        ----------

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        n_actions = 2
        ldf = self.ldf_.copy()
        filt__yhat1_giv_y1_z0 = (ldf['z'] == 0) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        filt__yhat1_giv_y1_z1 = (ldf['z'] == 1) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        ldf['A_ub'] = 0.0
        ldf.loc[filt__yhat1_giv_y1_z0, 'A_ub'] = 1
        ldf.loc[filt__yhat1_giv_y1_z1, 'A_ub'] = -1
        return ldf['A_ub']

    def _construct_reward__accuracy(self):
        """
        Constructs the reward function when the objective is accuracy.

        Parameters
        ----------

        Returns
        ------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = self.ldf_.copy()
        ldf['r'] = (ldf['yhat'] == ldf['y']).astype(float)
        c = -1 * ldf['r']  # Negative since maximizing not minimizing
        return c

    def _construct_reward__disp_imp_sub1(self):
        """
        Constructs the reward function for Disparate Impact Subproblem 1.
        Subproblem 1 is when we constrain P(yhat=1|z=0) >= P(yhat=1|z=1), in
        which case the reward penalizes giving the Z=0 group the positive
        prediction.

        Parameters
        ----------

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = self.ldf_.copy()
        filt__yhat1_giv_z0 = (ldf['z'] == 0) & (ldf['yhat'] == 1)
        filt__yhat1_giv_z1 = (ldf['z'] == 1) & (ldf['yhat'] == 1)
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_giv_z0, 'r'] = -2
        ldf.loc[filt__yhat1_giv_z1, 'r'] = 2
        c = -1 * ldf['r']  # Negative since maximizing not minimizing
        return c

    def _construct_reward__disp_imp_sub2(self):
        """
        Constructs the reward function for Disparate Impact Subproblem 2.
        Subproblem 2 is when we constrain P(yhat=1|z=1) >= P(yhat=1|z=0), in
        which case the reward penalizes giving the Z=1 group the positive
        prediction.

        Parameters
        ----------

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = self.ldf_.copy()
        filt__yhat1_giv_z0 = (ldf['z'] == 0) & (ldf['yhat'] == 1)
        filt__yhat1_giv_z1 = (ldf['z'] == 1) & (ldf['yhat'] == 1)
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_giv_z0, 'r'] = 2
        ldf.loc[filt__yhat1_giv_z1, 'r'] = -2
        c = -1 * ldf['r']  # Negative since maximizing not minimizing
        return c

    def _construct_reward__eq_opp_sub1(self):
        """
        Constructs the reward function for Equal Opportunity  Subproblem 1.
        Subproblem 1 is when we constrain P(yhat=1|y=1,z=0) >= P(yhat=1|y=1,z=1), in
        which case the reward penalizes giving the Z=0 group the positive
        prediction.

        Parameters
        ----------

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = self.ldf_.copy()
        filt__yhat1_giv_y1_z0 = (ldf['z'] == 0) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        filt__yhat1_giv_y1_z1 = (ldf['z'] == 1) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_giv_y1_z0, 'r'] = -5
        ldf.loc[filt__yhat1_giv_y1_z1, 'r'] = 5
        c = -1 * ldf['r']  # Negative since maximizing not minimizing
        return c

    def _construct_reward__eq_opp_sub2(self):
        """
        Constructs the reward function for Disparate Impact Subproblem 1.
        Subproblem 1 is when we constrain P(yhat=1|y=1,z=0) >= P(yhat=1|y=1,z=1), in
        which case the reward penalizes giving the Z=0 group the positive
        prediction.

        Parameters
        ----------

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = self.ldf_.copy()
        filt__yhat1_giv_y1_z0 = (ldf['z'] == 0) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        filt__yhat1_giv_y1_z1 = (ldf['z'] == 1) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_giv_y1_z0, 'r'] = 5
        ldf.loc[filt__yhat1_giv_y1_z1, 'r'] = -5
        c = -1 * ldf['r']  # Negative since maximizing not minimizing
        return c



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

    def predict(self, X):
        """
        Predicts `y` from `X`, then returns the optimal action for that value
        of `(X, y)`. Sort of like a two-step POMDP.

        I.e. here is some crude pseudocode reprsenting what's actually
        happening:
            ```
            y = predict(X)
            a = pi(X, y)
            return a
            ```
        Parameters
        ---------
        X : pandas.DataFrame
            Input data.

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
        df['y'] = self.clf.predict(X)
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


def _find_best_policies_from_multiple_subproblems(best_policies_best_rewards):
    """
    Parameters
    ----------
    best_policies_best_rewards : tuple<dict>
        Example:
        ```
        best_policies_best_rewards =
            {
                'policies': list(opt_pols_disp_imp_sub1_adult),
                'reward': np.round(opt_rew_disp_imp_sub1_adult, decimals=6)
            },
            {
                'policies': list(opt_pols_disp_imp_sub2_adult),
                'reward': np.round(opt_rew_disp_imp_sub2_adult, decimals=6)
            },
        )
        ```

    Returns
    -------
    best_of_best_pols : list<numpy.array>
        The unique list of optimal policies from all subproblems.
    best_of_best_reward : float
        The best reward of all subproblems.
    """
    rewards = [bpbr['reward'] for bpbr in best_policies_best_rewards]
    best_idx = np.argwhere(rewards == np.amax(rewards)).flatten().tolist()
    logging.debug(f"best_idx: {best_idx}")
    best_of_best_pols = []
    # For each subproblem index where the reward is the best reward
    for idx in best_idx:
        # Get all the policies from that subproblem (all have same reward)
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
        c, A_eq, b_eq, A_ub=None, b_ub=None, error_term=1e-12):
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
