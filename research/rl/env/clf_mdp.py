import itertools
import logging
import numbers
import numpy as np
import pandas as pd
from scipy.optimize import linprog


class ClassificationMDP:
    """
    Parameters
    ----------
    gamma : float, range, [0, 1)
        The discount factor. Should be close to zero since this is a
        classification mdp, but cannot be zero exactly otherwise the linear
        program won't converge.
    obj_set : ObjectiveSet
        The objective set.
    x_cols : list<str>
        The columns that are used in the state (along with `z` and `y`).

    Attributes
    ----------
    b_eq_ : numpy.array<float>
        A.k.a. "mu0". Initial state probabilities.
    A_eq_ : numpy.ndarray<float>, shape (len(df), 2*len(df))
        The state-action transition matrix.
    n_states_ : int
        Number of states.
    state_reducer_ : dict<str, dict<?, ?>>
        Specifies which state input columns and which values get replaced with
        default values. Used to reduce the state space by replacing infrequent
        state values with default values.
    reduced_state_df_ : pandas.DataFrame
        Index is state index, columns are features, mu0, and optimal policy
        actions.
    reduced_state_lookup_ : dict<tuple, int>
        Maps classification features to its MDP state index.
    ldf_ : pandas.DataFrame
        "Lambda dataframe". One row for each state and action combination.
        Columns are **x_cols, z, y, yhat.
    b_ub_dem_par_ : np.array<float>, len(n_states_)
        The uppber bound `b` for Demographic Parity Split1 and Split2.
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

    def __init__(self, gamma, obj_set, x_cols):
        self.gamma = gamma
        self.obj_set = obj_set
        self.x_cols = x_cols
        self.state_reducer_ = {}
        self.reduced_state_df_ = None
        self.reduced_state_lookup_ = None
        self.n_states = None
        self.A_eq_ = None
        self.b_eq_ = None
        self.ldf_ = None
        self.opt_problems_ = None

    def fit(self, reward_weights, clf_df, min_freq_fill_pct=0, restrict_y=True):
        """
        Parameters
        ----------
        reward_weights : dict<str, float>
            Keys are objective identifiers. Values are their respective reward
            weights.
        clf_df : pandas.DataFrame
            Classification dataset. Required columns:
                'z' : int. Binary protected attribute.
                'y' : int. Binary target variable.

        min_freq_fill_pct, float, range[0, 1), default 0
            Minimum frequency for each input variable to not get replaced by a
            default value.
        restrict_y : bool, deafult True
            If True, policy must have same action for any x,z combo, regardless
            of y.

        Sets Attributes
        ---------------
        b_eq_
        A_eq_
        state_reducer_
        reduced_state_lookup_
        reduced_state_df_
        n_states_
        ldf_
        opt_problems_

        Returns
        -------
        self
        """
        clf_df = clf_df.copy()

        # Set the min frequency replacement values to reduce state
        # min_freq_fill_pcts : dict<str, (float, range [0,1], ?)>, default {}
        #     Dictionary with col_name -> (min_freq_pct, default_val)
        #     that specifies the minimum frequency to replace a each input
        #     attribute with a default value.  It's a hacky equivalent to what
        #     the `min_frequency` parameter does for the scikitlearn
        #     OneHotEncoder.
        min_freq_fill_pcts = {}
        for x in self.x_cols:
            min_freq_fill_pcts[x] = (
                min_freq_fill_pct,
                -555,  # Distinct value for too infrequent values.
                # clf_df[x].value_counts().sort_values().index[-1],  # most freq.
            )
        logging.debug('\nmin_freq_fill_pcts:')
        logging.debug(min_freq_fill_pcts)

        # Generate the state_reducer_ object that replaces infrequent state
        # values with defaults.
        logging.debug('\nFitting state_reducer_ ...')
        for x in min_freq_fill_pcts.keys():
            min_freq, default_val= min_freq_fill_pcts[x]
            freq = clf_df.groupby(x).size() / len(clf_df)
            for x_val in (freq[freq < min_freq]).index:
                try:
                    if x not in self.state_reducer_:
                        self.state_reducer_[x] = {}

                    self.state_reducer_[x][x_val] = default_val
                    clf_df.loc[clf_df[x] == x_val, x] = default_val
                except KeyError:
                    continue

        logging.debug(f"\nstate_reducer_: \n{self.state_reducer_}")
        # Generate the reduced_state_df_ and the X,y -> state mapping
        self.reduced_state_df_ = (
            clf_df
            .groupby(self.x_cols+['z', 'y']).size()
            .reset_index().rename(columns={0: 'count'})
        )

        self.reduced_state_lookup_ = {}
        state_counter = 0
        for idx, row in self.reduced_state_df_.iloc[:, :-1].iterrows():
            self.reduced_state_lookup_[tuple(row)] = state_counter
            state_counter += 1

        # Cache n_states since frequently used in computations
        self.n_states_ = state_counter

        # Compute `mu0` (initial state probabilities)
        logging.debug('Computing b_eq ...')
        self.reduced_state_df_['mu0'] = self.reduced_state_df_['count'] / self.reduced_state_df_['count'].sum()
        self.reduced_state_df_ = self.reduced_state_df_.drop(columns='count')
        logging.debug(f"n_states: {len(self.reduced_state_df_)}")

        # Generate the lambda dataframe (state-action indexed)
        logging.debug('Generating the lambda dataframe ...')
        self.ldf_ = self._generate_lambda_linear_equations(self.reduced_state_df_)

        # Compute b_eq, which consists of two parts:
        #   1: mu0
        #   2: zeros (for action equality for same y-values)
        self.b_eq_ = self.reduced_state_df_['mu0']
        if restrict_y:
            self.b_eq_ = np.concatenate([
                self.b_eq_,
                np.zeros(self.ldf_.groupby(self.x_cols+['z', 'yhat']).size().shape[0]),  # action equality for y
            ])

        # Compute transition matrix linear equations `A_eq`, which consists of
        # two parts:
        #   1: Transition matrix
        #   2: Action equality for same y-values
        logging.debug('Computing transition matrix linear equations A_eq ...')
        self.A_eq_ = self._compute_A_eq(
            mu0=self.reduced_state_df_['mu0'],
            ldf=self.ldf_,
            x_cols=self.x_cols,
            restrict_y=restrict_y,
        )

        logging.debug('Fitting objectives ...')
        n_primary_constr = len(self.reduced_state_df_['mu0'])
        A_eq = np.concatenate([
            self.A_eq_[0:n_primary_constr],
            self.A_eq_[n_primary_constr+0:n_primary_constr+2],
        ])
        b_eq = np.concatenate([
            self.b_eq_[0:n_primary_constr],
            self.b_eq_[n_primary_constr+0:n_primary_constr+2],
        ])
        self.obj_set.fit(
            reward_weights=reward_weights,
            ldf=self.ldf_,
            A_eq=self.A_eq_,
            b_eq=self.b_eq_,
        )
        self.opt_problems_ = self.obj_set.opt_problems_

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
            opt_pols, opt_rew = _find_all_solutions_lp(
                n_states=self.n_states_,
                c=subprob.c,
                A_eq=subprob.A_eq,
                b_eq=subprob.b_eq,
                A_ub=subprob.A_ub,
                b_ub=subprob.b_ub,
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

        # Append optimal policy actions to reduced_state_df_ attribute
        for i, pi in enumerate(opt_pols):
            self.reduced_state_df_[f"pi_{i}"] = pi

        return opt_pols

    def _compute_A_eq(self, mu0, ldf, x_cols, restrict_y):
        """
        Constructs two sets of linear equation constraints.

        1st set: transition matrix
        --------------------------
        Constructs a set of linear equations representing the state-action
        transition probabilities  where all "states" (classification dataset
        samples) have the initial state probability, regardless of the action
        taken.

        2nd set: require equal actions for same `y`
        -------------------------------------------
        # Construct constraints that require equal actions for the same `y`
        # value. This constraint helps the optimizer learn policies that are
        # robust to incorrect predictions of `y`. This was causing problems
        # when trying to optimize for fairness constraints like EqOpp that
        # require knowledge of the `y` value.

        Parameters
        ----------
        mu0 : 1-D array
            Initial state probabilities.
        ldf : pd.DataFrame
            "Lambda dataframe". One row for each state and action combination.
        x_cols : list<str>
            The columns that are used in the state (along with `z` and `y`).
        restrict_y : bool
            If True, policy must have same action for any x,z combo, regardless
            of y.

        Returns
        -------
        A_eq_
        """
        ldf_mu0 = ldf.copy()
        ldf = ldf.copy().drop(columns='mu0')

        # Construct constraints that correspond to transition matrix.
        n_states = len(mu0)
        n_actions = 2
        A_eq = np.zeros((n_states, n_states*n_actions))
        for s in range(n_states):
            for sp in range(n_states):
                for a in range(n_actions):
                    if s == sp:
                        A_eq[s][sp*n_actions+a] = 1 - self.gamma*mu0[sp]
                    else:
                        A_eq[s][sp*n_actions+a] = 0 - self.gamma*mu0[sp]

        if restrict_y:
            # Construct constraints that require equal actions for the same `y`
            # value. This constraint helps the optimizer learn policies that are
            # robust to incorrect predictions of `y`. This was causing problems
            # when trying to optimize for fairness constraints like EqOpp that
            # require knowledge of the `y` value.

            # For each, x, a combination:
            #   Add constraint that x,y0,a == x,y1,a
            n_constr = ldf.groupby(x_cols+['z', 'yhat']).size().shape[0]
            A_eq2 = np.zeros((n_constr, len(ldf)))

            group_i = 0
            for cols, group in ldf.groupby(x_cols+['z', 'yhat']):
                assert len(group) <= 2

                if len(group) == 1:
                    continue

                constr = np.zeros_like(ldf.index)
                locy0 = group.index[0]
                locy1 = group.index[1]
                _group = group[x_cols+['z']]
                # Find the mu0 values for each x, a group
                locy0_mu0 = ldf_mu0[(ldf_mu0[x_cols+['z', 'y']] == list(_group.iloc[0].values)+[0]).sum(axis=1) == len(x_cols)+2]['mu0']
                locy1_mu0 = ldf_mu0[(ldf_mu0[x_cols+['z', 'y']] == list(_group.iloc[0].values)+[1]).sum(axis=1) == len(x_cols)+2]['mu0']
                assert len(locy0_mu0) == 2
                assert len(locy1_mu0) == 2
                locy0_mu0 = locy0_mu0.values[0]
                locy1_mu0 = locy1_mu0.values[0]
                constr[locy0] = 1 * locy1_mu0
                constr[locy1] = -1 * locy0_mu0

                A_eq2[group_i] = constr
                group_i += 1

            # Combine the two constraint matrices
            A_eq = np.concatenate([A_eq, A_eq2])

        return A_eq

    def _generate_lambda_linear_equations(self, reduced_state_df):
        """
        TODO
        """
        state_df = reduced_state_df.copy()
        # Set every two rows the same. One for each action.
        ldf = pd.concat([state_df, state_df], axis=0).reset_index(drop=True)
        ldf = ldf.sort_values(list(ldf.columns))
        yhat = np.zeros(len(ldf), dtype=int)
        yhat[1::2] = 1  # Makes 'a' 0, 1 repeating sequence
        ldf['yhat'] = yhat
        ldf = ldf.reset_index(drop=True)
        return ldf


def _find_best_policies_from_multiple_opt_problems(best_policies_best_rewards):
    """
    Parameters
    ----------
    best_policies_best_rewards : tuple<dict>
        Example:
        ```
        best_policies_best_rewards =
            {
                'policies': list(opt_pols_dem_par_split1_adult),
                'reward': np.round(opt_rew_dem_par_split1_adult, decimals=6)
            },
            {
                'policies': list(opt_pols_dem_par_split2_adult),
                'reward': np.round(opt_rew_dem_par_split2_adult, decimals=6)
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
        n_states, c, A_eq, b_eq, A_ub=None, b_ub=None, error_term=1e-12, skip_error_terms=False, method='highs'):
    """
    Wrapper around scipy.optimize.linprog that finds ALL optimal solutions
    by iteratively solving the LP problem after adding/subtracting an "error"
    term to each objective component.

    Parameters
    ----------
    n_states = int
        Number of states.
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
    n_actions = 2

    if skip_error_terms:

        if A_ub is None or len(A_ub) == 0:
            assert(b_ub is None or len(b_ub) == 0)
            res = linprog(c, A_eq=A_eq, b_eq=b_eq)
        else:
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
