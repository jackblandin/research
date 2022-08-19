# Core modules
import logging.config

# 3rd party modules
import gym
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from gym.spaces import Discrete, Tuple
from scipy.optimize import linprog


class DiscreteMDP(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, state_dims, action_dims, obs_dims,
            max_steps_per_episode, verbose=False, args={}):
        """
        Generic environment for a discrete MDP. This class includes helper
        methods for computing transitions, observations, and rewards. Can also
        create a POMDP if desired. Also has a an optimal policy helper method
        that uses linear programming.

        Parameters
        ----------
        state_dims : tuple<int>
            Dimensions of each state component.
        action_dims : tuple<int>
            Dimensions of each action component.
        obs_dims : tuple<int>
            Dimensions of each observation component.
        max_steps_per_episode : int
            Maximum allowed steps per episode. This will define how long an
            episode lasts, since the environment does not end otherwise.
        verbose : bool, default False
            If True, log current state on each timestep.
        args : dict, default {}
            Any extra parameters needed for specific environments. Useful so
            that child classes don't need to modify their __init__ method when
            they want additional arguments.

        Attributes
        ----------
        n_states : int
            Number of possible states computed as `np.prod(*state_dims)`.
        n_actions : int
            Number of possible actions computed as `np.prod(*action_dims)`.
        mu0 : np.array(n_states)
            Initial state distribution. Computed from
            `_init_state_probability()` injected method.
        T : np.ndarray (n_states, n_actions, n_states)
            3-dim matrix where index [s][a][s'] represents the probability of
            transitioning from state s to state s' after taking action a.
            Computed from `_transition_probability()` injected method.
        Osaso : np.ndarray (n_states, n_actions, n_states, n_obs)
            4-dim matrix where vector [s][a][s'] represents the probabilities
            of each observation given action a is taken from state s and it
            transitions to s'. Computed from `_observation_probability()`
            injected method.
        Rsas : np.ndarray (n_states, n_actions, n_states)
            3-dim matrix where index [s][a][s'] represents the reward for
            transitioning from state s to state s' after taking action a.
            Computed from `_reward_sas()` injected method.
        cur_episode : int
            Current episode as a count.
        action_episode_memory : list<<list<int>>
            History of actions taken in each episode.
        observation_episode_memory : list<list<int>>
            History of observations observed in each episode.
        reward_episode_memory : list<list<float>>
            History of rewards observed in each episode.
        cur_step : int
            Current timestep in episode, as a count.
        n_actions : int
            Number of possible actions.
        n_obs : int
            Number of possible observations.
        cur_state : int
            Current state.

        Private attributes
        ------------------
        _state_to_feats : np.ndarray, shape(n_states, len(state_dims))
            Feature values lookup by state index.
        _feats_to_state : object<str, int>
            State index lookup by feature values.
        _obs_to_feats : np.ndarray, shape(n_obs, len(obs_dims))
            Feature values lookup by observation index.
        _feats_to_obs : object<str, int>
            Observation index lookup by feature values.

        Examples
        --------
        >>> import gym
        >>> from research.rl.env.helpers import play_n_episodes
        >>> from reserach.rl.env.discrete_mdp import DiscreteMDP
        >>> args = {'max_steps_per_episode': 100}
        >>> env = DiscreteMDP(Ospo=...,
                              T=...
                              max_steps_per_episode=1_000,
                              Rsp=...)
        >>> model = TODO
        >>> play_n_episodes(env, model, n=1)
        """
        self.__version__ = "0.0.1"
        logging.info("DiscreteMDP - Version {}".format(self.__version__))

        self.state_dims = state_dims
        self.action_dims = action_dims
        self.obs_dims = obs_dims
        self.args = args

        # These need to come before the _construct methods
        self.n_states = np.prod(self.state_dims)
        self.n_actions = np.prod(self.action_dims)
        self.n_obs = np.prod(self.obs_dims)

        self.max_steps_per_episode = max_steps_per_episode
        self.verbose = verbose
        self.cur_episode = -1  # Set to -1 b/c reset() adds 1 to episode
        self.action_episode_memory = []
        self.observation_episode_memory = []
        self.reward_episode_memory = []
        self.cur_step = 0

        # Build state hash
        self._construct_state_feat_lookups()
        # Build observation hash
        self._construct_obs_feat_lookups()

        # Build initial state vector mu0 from injected method
        self.mu0 = np.array([
            self._init_state_probability(s) for s in range(self.n_states)
        ])

        # Build transition matrix T from injected method
        self.T = np.array([
            [self._transition_probability(s, a) for a in range(self.n_actions)]
            for s in range(self.n_states)])

        # Build observation matrix Osaso from injected method
        self.Osaso = np.array([
            np.array([
                np.array([
                    self._observation_probability(s, a, sp) for sp in range(self.n_states)
                ]) for a in range(self.n_actions)
            ]) for s in range(self.n_states)
        ])

        # Build reward matrix Rsas from injected method.
        self.Rsas = np.array([
            np.array([
                np.array([
                    self._reward_sas(s, a, sp) for sp in range(self.n_states)
                    ]) for a in range(self.n_actions)
            ]) for s in range(self.n_states)
        ])

        # Check all O[s][a][s'][:] probs sum to 1
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for sp in range(self.n_states):
                    assert self.Osaso[s][a][sp].sum() == 1

        # Check all T[s][a][:] probs sum to 1
        for s in range(self.n_states):
            for a in range(self.n_actions):
                try:
                    assert (abs(self.T[s][a].sum() - 1) < 1e6)
                except:
                    _sum = self.T[s][a].sum()
                    log_msg = f'self.T[{s}][{a}].sum() = {self.T[s][a].sum()}'
                    err_msg = f'T[s={s}][a={a}] sums to {_sum:.3f}, not 1.'
                    s_feats = self._state_to_feats[s]
                    for sp in range(0, self.n_states):
                        if self.T[s][a][sp] > 0:
                            sp_feats = self._state_to_feats[sp]
                            display(f'T{s_feats}[{a}]{sp_feats}] = {self.T[s][a][sp]}')
                    logging.debug(log_msg)
                    raise AssertionError(err_msg)

        # Check mu0.shape matches T.shape
        assert self.mu0.shape[0] == self.T.shape[0]
        # Check T.shape == Rsas.shape
        assert self.T.shape == self.Rsas.shape
        # Check Osaso matches shape of T
        assert self.Osaso.shape[0:3] == self.T.shape


        # Reset
        self.reset()

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : int
            Action to take.

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob : list
                A list of ones or zeros which together represent the state of
                the environment.
            reward : float
                Amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            done : bool
                Whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info : dict
                Diagnostic information useful for debugging. It can sometimes
                be useful for learning (for example, it might contain the raw
                probabilities behind the environment's last state change).
                However, official evaluations of your agent are not allowed to
                use this for learning.
        """
        done = self.cur_step >= self.max_steps_per_episode
        if done:
            raise RuntimeError("Episode is done")

        prev_state = self.cur_state

        self.cur_step += 1

        # Compute new state based on previous state and action
        new_state = self._take_action(action)

        # Compute reward value based on new state
        reward = self._get_reward(s=prev_state, a=action, sp=new_state)

        # Update current state to new state
        self.cur_state = new_state

        # Compute observation from current state
        ob = self._get_obs(s=prev_state, a=action, sp=new_state)

        # Update action, observation and reward histories
        self.action_episode_memory[self.cur_episode].append(action)
        self.observation_episode_memory[self.cur_episode].append(ob)
        self.reward_episode_memory[self.cur_episode].append(reward)

        # Recompute done since action may have modified it
        done = self.cur_step >= self.max_steps_per_episode

        if self.verbose:
            logging.info('\t' + self.render_state(self.cur_state))

        return ob, reward, done, {}

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        object
            The initial observation of the space.
        """
        self.cur_step = 0
        self.cur_episode += 1
        self.cur_state = self._sample_initial_state()
        ob = self._get_obs(s=None, a=None, sp=self.cur_state)

        # if self.cur_episode > 0:
        self.action_episode_memory.append([])
        self.observation_episode_memory.append([])
        self.reward_episode_memory.append([])

        # Update action, observation and reward histories
        self.action_episode_memory[self.cur_episode].append(-1)
        self.observation_episode_memory[self.cur_episode].append(ob)
        self.reward_episode_memory[self.cur_episode].append(-1)

        if self.verbose:
            logging.info(f'Episode {self.cur_episode}')

        if self.verbose:
            logging.info('\t' + self.render_state(self.cur_state))

        return ob

    def render(self, mode='human'):
        return

    def close(self):
        pass

    def render_policy(self, pi, state_labels):
        """
        Returns a policy in a DataFrame format.

        Parameters
        ----------
        pi : np.array<int>, len(env.n_obs)
            Policy to render.
        state_labels : array, shape == state_dims
            Labels for each state dimension

        Returns
        -------
        pd.DataFrame
            Policy in dataframe format.
        """
        pi_df = pd.DataFrame(data=self._state_to_feats, columns=state_labels)
        pi_df['mu0'] = self.mu0
        pi_df['a'] = pi
        return pi_df

    @abstractmethod
    def render_state(self, s):
        """
        String representation of a particular state.

        Parameters
        ----------
        s : int
            State index.

        Returns
        -------
        str
            String representation in human readable format.
        """
        pass

    def render_env_history(self):
        """
        Returns the env observation, action, and reward history as a pandas df.

        Returns
        -------
        pandas.DataFrame()
            columns: episode, timestep, *features, action, reward
        """
        n_eps = len(self.observation_episode_memory[1:])
        n_steps = len(self.observation_episode_memory[1])
        metrics_by_ep = np.zeros(n_eps)
        feat_cols = ['z', 'y0', 'y1', 'c']
        df = pd.DataFrame([],
                columns=['episode', 'timestep', *feat_cols, 'a', 'r'])

        for ep in range(n_eps):
            obss = self.observation_episode_memory[ep+1]
            acts = self.action_episode_memory[ep+1]
            rewards = self.reward_episode_memory[ep+1]
            feats = np.ndarray((len(obss), len(self.state_dims)))
            for i, s in enumerate(obss):
                feats[i] = np.array(self._state_to_feats[s])
            ep_df = pd.DataFrame(feats, columns=feat_cols)
            ep_df['a'] = acts[1:] + [np.nan]  # Actions are offset by 1 step
            ep_df['episode'] = ep
            ep_df['timestep'] = np.arange(n_steps)
            ep_df['r'] = rewards[1:] + [np.nan]  # Rewards are offset by 1 step
            df = pd.concat([df, ep_df])

        df = df.reset_index(drop=True)

        return df

    ###################
    # Private methods #
    ###################

    def _construct_Osaso_from_Ospo(self, Ospo):
        """
        Constructs observation probability matrix O[s][a][s'][o] from a
        O[s'][o]. (Sometime's it's easier to construct an observation function
        based on state alone).

        Parameters
        ----------
        Ospo : np.ndarray(n_states, n_obs)
            O[s'][o] reward function.

        Returns
        -------
        np.ndarray (n_states, n_actions, n_states, n_obs)
            Osaso[s][a][s'][o] format of Ospo[s'][o].

        """
        Osaso = np.zeros((self.n_states, self.n_actions, self.n_states,
            self.n_obs))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for sp in range(self.n_states):
                    for o in range(self.n_obs):
                        Osaso[s][a][sp][o] = Ospo[sp][o]
        return Osaso

    def _construct_state_feat_lookups(self):
        """
        Assigns an integer to each possible state. The `_state_to_feats`
        attribute will provide the mapping of state index to state values.
        Similarly, the `_feats_to_state` attribute will provide the reverse
        lookup.
        Returns
        -------
        None

        Sets
        ----
        _state_to_feats : np.ndarray, shape(n_states, len(state_dims))
            Feature values lookup by state index.
        _feats_to_state : object<str, int>
            State index lookup by feature values.
        """
        # construction variables
        _state_to_feats = np.zeros((self.n_states, len(self.state_dims)),
                dtype=int)
        _feats_to_state = {}
        # temp variables
        state_comp_cur_idxs = np.zeros_like(self.state_dims)
        furth_right_non_max_idx = len(self.state_dims) - 1
        state_idx = 0
        done = False

        # Iterate over every permutation of state features
        while not done:
            # Set _state_to_feats and _feats_to_state
            #   with current vals of temp vars.
            #   state_comp_cur_idxs e.g. [3, 0, 2] if 3 state components.
            #   State component 0 is at index 3, ...
            for state_comp_idx, state_comp_val in enumerate(
                    state_comp_cur_idxs):
                _state_to_feats[state_idx][state_comp_idx] = state_comp_val
                _feats_to_state[
                        self._hash_features(state_comp_cur_idxs)] = state_idx

            # Increment/decrement temp vars

            # Increment state_idx since it's been used (above).
            state_idx += 1

            # Starting from the right, find the first index val that's < it's
            # respective dim size, increment it by 1, then reset all index vals
            # to the right of it to zero.
            done = True
            for idx in range(len(state_comp_cur_idxs)-1, -1, -1):
                if state_comp_cur_idxs[idx] < self.state_dims[idx]-1:
                    state_comp_cur_idxs[idx] += 1
                    for comp_idx in range(idx+1, len(self.state_dims)):
                        state_comp_cur_idxs[comp_idx] = 0
                    done = False
                    break

        self._state_to_feats = _state_to_feats
        self._feats_to_state = _feats_to_state


    def _construct_obs_feat_lookups(self):
        """
        Assigns an integer to each possible observation. The `_obs_to_feats`
        attribute will provide the mapping of obs index to state values.
        Similarly, the `_feats_to_obs` attribute will provide the reverse
        lookup.

        Returns
        -------
        None

        Sets
        ----
        _obs_to_feats : np.ndarray, shape(n_states, len(state_dims))
            Feature values lookup by obs index.
        _feats_to_obs : object<str, int>
            Obs index lookup by feature values.
        """
        # construction variables
        _obs_to_feats = np.zeros((self.n_obs, len(self.obs_dims)),
                dtype=int)
        _feats_to_obs = {}
        # temp variables
        obs_comp_cur_idxs = np.zeros_like(self.obs_dims)
        furth_right_non_max_idx = len(self.obs_dims) - 1
        obs_idx = 0
        done = False

        # Iterate over every permutation of obs features
        while not done:
            # Set _obs_to_feats and _feats_to_obs
            #   with current vals of temp vars.
            #   obs_comp_cur_idxs e.g. [3, 0, 2] if 3 obs components.
            #   obs component 0 is at index 3, ...
            for obs_comp_idx, obs_comp_val in enumerate(
                    obs_comp_cur_idxs):
                _obs_to_feats[obs_idx][obs_comp_idx] = obs_comp_val
                _feats_to_obs[
                        self._hash_features(obs_comp_cur_idxs)] = obs_idx

            # Increment/decrement temp vars

            # Increment obs_idx since it's been used (above).
            obs_idx += 1

            # Starting from the right, find the first index val that's < it's
            # respective dim size, increment it by 1, then reset all index vals
            # to the right of it to zero.
            done = True
            for idx in range(len(obs_comp_cur_idxs)-1, -1, -1):
                if obs_comp_cur_idxs[idx] < self.obs_dims[idx]-1:
                    obs_comp_cur_idxs[idx] += 1
                    for comp_idx in range(idx+1, len(self.obs_dims)):
                        obs_comp_cur_idxs[comp_idx] = 0
                    done = False
                    break

        self._obs_to_feats = _obs_to_feats
        self._feats_to_obs = _feats_to_obs

        return None

    def _hash_features(self, features):
        """
        Produces the hash value for the provided features. Used to get the
        integer state value associated with a particular set of features.

        Parameters
        ----------
        features : tuple, shape(state_dims)
            The state in features format.

        Returns
        -------
        str
            Hash value for the corresponding state.
        """
        return ''.join(np.array(features).astype(str))

    @abstractmethod
    def _init_state_probability(self, s):
        """
        Returns the probabily of starting in state s.

        Parameters
        ----------
        s : int
            Initial state index.

        Returns
        -------
        float, range(0,1)
            Initial state probability.
        """
        pass

    @abstractmethod
    def _transition_probability(self, s, a):
        """
        Returns the probabilities of transitioning into all next states after
        taking action a in state s.

        Parameters
        ----------
        s : int
            Initial state index.
        a : int
            Action index.

        Returns
        -------
        np.array<float>, shape(n_states)
            Transition probabilities.
        """
        pass

    @abstractmethod
    def _observation_probability(self, s, a, sp):
        """
        Returns the probabilities of observing all observations after taking
        action a in state s and transitioning into state sp. Used to construct
        `Osaso`.

        Parameters
        ----------
        s : int
            Initial state index.
        a : int
            Action index.
        sp : int
            New state index.

        Returns
        -------
        np.array<float>, shape(n_obs)
            Observation probabilities.
        """
        pass

    @abstractmethod
    def _reward_sas(self, s, a, sp):
        """
        Returns the reward value obtained after taking action `a` in state `s`
        and transitioning into new state `sp`. Used to construct `Rsas`.

        Parameters
        ----------
        s : int
            Initial state index.
        a : int
            Action index.
        sp : int
            New state index.

        Returns
        -------
        float
            Reward value.
        """
        pass

    def _sample_initial_state(self):
        """
        Samples initial state from mu0.

        Returns
        -------
        int
            Initial state index.
        """
        s0 = np.random.choice(np.arange(self.n_states), p=self.mu0)
        return s0


    def _take_action(self, action):
        """
        How to change the environment when taking an action.

        Parameters
        ----------
        action : int
            Action.

        Returns
        -------
        int
            New state after taking action.
        """
        # Get transition probabilities for all potential next state values
        trans_probs = self.T[self.cur_state, action]

        # Generate an array of next state options to choose from
        next_state_options = np.linspace(0, self.n_states-1, self.n_states,
                                         dtype=int)

        # Sample from new state options based on the transition probabilities
        new_state = np.random.choice(next_state_options, p=trans_probs)

        return new_state

    def _get_reward(self, s, a, sp):
        """
        Returns the reward based on s, a, s'.

        Parameters
        ----------
        s : int
            Index for state s.
        a : int
            Index for action a.
        sp : int
            Index for state s'.

        Returns
        -------
        float
            Reward(s,a,s').
        """
        return self.Rsas[s,a,sp]

    def _get_obs(self, s, a, sp):
        """
        Obtain the observation for the previous state, previous action, and new
        state. If the environment is fully observable, the state is returned
        directly. All parameters are required except for initial state which
        should have a non-negative sp value, but s and a are not present.

        Parameters
        ----------
        s : int
            State s index.
        a : int
            Action a index
        sp : int
            s' index

        Returns
        -------
        int
            Observation index.
        """
        if s is None and a is None:
            # TODO: this is hacky since we just sample a random previous state
            # and action. Think # of something better.
            # Generate an array of next state options to choose from
            action_options = np.linspace(0, self.n_actions-1, self.n_actions,
                                         dtype=int)
            # Sample from all states
            s = np.random.choice(np.arange(self.n_states),
                    p=(np.ones(self.n_states)/self.n_states))

            # Sample from all actions
            a = np.random.choice(np.arange(self.n_actions),
                    p=(np.ones(self.n_actions)/self.n_actions))

        # Get probabilities for all potential observations
        obs_probs = self.Osaso[s][a][sp]

        # Generate an array of observation options to choose from
        obs_options = np.linspace(0, self.n_obs-1, self.n_obs, dtype=int)

        # Sample from observation options using observation probabilities
        obs = np.random.choice(obs_options, p=obs_probs)

        return obs


def compute_optimal_policy(env, gamma):
    """
    Computes optimal policy of a DiscreteMDP using LP dual.

    Scipy solver with min_x c^T.dot(x) s.t. A.dot(x) = b.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

    Used slide 32 from following to derive LP program for solving MDP:
    https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/mdps-exact-methods.pdf

    Parameters
    ----------
    env : DiscreteMDP
        MDP environment.
    gamma : float, [0,1]
        Discount factor.

    Returns
    -------
    np.array<int>, len(env.n_obs)
        Optimal policy.
    """
    # Construct dual_A[s][s'*|A|+a]
    dual_A = np.zeros((env.n_states, env.n_states*env.n_actions))
    for s in range(env.n_states):
        for sp in range(env.n_states):
            for a in range(env.n_actions):
                if s == sp:
                    dual_A[s][sp*env.n_actions+a] = 1 - gamma*env.T[sp][a][s]
                else:
                    dual_A[s][sp*env.n_actions+a] = 0 - gamma*env.T[sp][a][s]

    # Construct dual_c
    dual_c = np.zeros(env.n_states*env.n_actions)
    for s in range(env.n_states):
        for a in range(env.n_actions):
            _sum = 0
            for sp in range(env.n_states):
                _sum += (env.T[s][a][sp] * env.Rsas[s][a][sp])
            dual_c[s*env.n_actions+a] = _sum

    dual_c = -1*dual_c  # Multiply by -1 since maximizing

    # Construct dual_b = mu0
    dual_b = env.mu0

    # Solve linear program
    res = linprog(dual_c, A_eq=dual_A, b_eq=dual_b)

    # Compute optimal policy (only one)
    # Below works, leaving here in case I mess it up with multiple optimal
    # policy computations.
    # pi_opt = np.zeros(env.n_states, dtype=int)
    # for s in range(env.n_states):
    #     start_idx = s*env.n_actions
    #     end_idx = s*env.n_actions+env.n_actions
    #     pi_opt[s] = res.x[start_idx:end_idx].argmax()

    # Compute all optimal policies.
    pi_opts = [np.zeros(env.n_states, dtype=int)]
    for s in range(env.n_states):
        start_idx = s*env.n_actions
        end_idx = s*env.n_actions+env.n_actions
        state_idxs = res.x[start_idx:end_idx]
        best_a = np.flatnonzero(state_idxs == np.max(state_idxs))
        # Use index in enumeration since we're modifying the list itself.
        for pi_i in range(len(pi_opts)):
            for i, a in enumerate(best_a):
                if i > 0:
                    pi_copy = pi_opts[pi_i].copy()
                    pi_copy[s] = a
                    pi_opts.append(pi_copy)
                else:
                    pi_opts[pi_i][s] = a
            # n_best_acts = len(best_a)
            # if n_best_acts > 1:
            #     print('s', s)
            #     print(f'{n_best_acts} found')
            #     print('state_idxs', state_idxs)
            #     print('best_a', best_a)

    return pi_opts
