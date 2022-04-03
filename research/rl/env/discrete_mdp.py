# Core modules
import logging.config

# 3rd party modules
import gym
import numpy as np
from abc import ABC, abstractmethod
from gym.spaces import Discrete, Tuple
from scipy.optimize import linprog


class DiscreteMDP(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, mu0, T, max_steps_per_episode, Rsas=None, Rsp=None,
            Osaso=None, Ospo=None, verbose=False):
        """
        Generic environment for a discrete MDP. This class includes helper
        methods for computing transitions, observations, and rewards. Can also
        create a POMDP if desired. Also has a an optimal policy helper method
        that uses linear programming.

        Parameters
        ----------
        mu0 : np.array(n_states)
            Initial state distribution.
        T : np.ndarray (n_states, n_actions, n_states)
            3-dim matrix where index [s][a][s'] represents the probability of
            transitioning from state s to state s' after taking action a.
        max_steps_per_episode : int
            Maximum allowed steps per episode. This will define how long an
            episode lasts, since the environment does not end otherwise.
        Rsas : np.ndarray (n_states, n_actions, n_states) : optional
            3-dim matrix where index [s][a][s'] represents the reward for
            transitioning from state s to state s' after taking action a.
            Required unless Rsp present.
        Rsp : np.ndarray (n_states) : optional
            1-dim vector where index [s'] represents the reward for
            R[s][a][s']. Required unless Rsas present.
        Osaso : np.ndarray (n_states, n_actions, n_states, n_obs) : optional
            4-dim matrix where vector [s][a][s'] represents the probabilities
            of each observation given action a is taken from state s and it
            transitions to s'. Required unless Ospo present.
        Ospo : np.ndarray (n_states) : optional
            2-dim vector where index [s'][o] represents the probability that
            observtion o will be observed in state s'. Required unless Osasoo
            present.
        verbose : bool, default False
            If True, log current state on each timestep.

        Attributes
        ----------
        action_space : gym.spaces.Discrete
            Action space.
        observation_space : gym.spaces.Discrete
            Observation space.
        n_states : int
            Number of possible states.
        cur_episode : int
            Current episode as a count.
        action_episode_memory : list<<list<int>>
            History of actions taken in each episode.
        observation_episode_memory : list<list<int>>
            History of observations observed in each episode.
        reward_episode_memory : list<list<int, float>>
            History of rewards observed in each episode.
        cur_step : int
            Current timestep in episode, as a count.
        n_actions : int
            Number of possible actions.
        n_obs : int
            Number of possible observations.
        cur_state : int
            Current state.

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

        self.mu0 = mu0
        self.T = T
        self.Rsas = Rsas

        # These need to come before the _construct methods
        self.action_space = Discrete(T.shape[1])
        if Osaso is not None:
            self.n_obs = Osaso.shape[3]
        else:
            self.n_obs = Ospo.shape[1]
        self.n_states = T.shape[0]
        self.n_actions = self.action_space.n
        self.observation_space = Discrete(self.n_obs)

        if self.Rsas is None and Rsp is not None:
            self.Rsas = self._construct_Rsas_from_Rsp(Rsp)
        elif Rsas is None:
            raise ValueError('Either Rsas or Rsp must be present.')

        self.Osaso = Osaso
        if self.Osaso is None and Ospo is not None:
            self.Osaso = self._construct_Osaso_from_Ospo(Ospo)
        elif Osaso is None:
            raise ValueError('Either Osaso or Ospo must be present.')

        # Check all O[s][a][s'][:] probs sum to 1
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for sp in range(self.n_states):
                    assert self.Osaso[s][a][sp].sum() == 1

        # Check all T[s][a][:] probs sum to 1
        for s in range(self.n_states):
            for a in range(self.n_actions):
                assert T[s][a].sum() == 1

        # Check mu0.shape matches T.shape
        assert self.mu0.shape[0] == self.T.shape[0]
        # Check T.shape == Rsas.shape
        assert T.shape == self.Rsas.shape
        # Check Osaso matches shape of T
        assert self.Osaso.shape[0:3] == T.shape

        self.max_steps_per_episode = max_steps_per_episode
        self.verbose = verbose
        self.cur_episode = -1  # Set to -1 b/c reset() adds 1 to episode
        self.action_episode_memory = []
        self.observation_episode_memory = []
        self.reward_episode_memory = []
        self.cur_step = 0

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
            episode_over : bool
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
        self.action_episode_memory.append([])
        self.observation_episode_memory.append([])
        self.reward_episode_memory.append([])
        obs = self._get_obs(s=None, a=None, sp=self.cur_state)

        if self.verbose:
            logging.info(f'Episode {self.cur_episode}')

        if self.verbose:
            logging.info('\t' + self.render_state(self.cur_state))

        return obs

    def render(self, mode='human'):
        return

    def close(self):
        pass

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

    ###################
    # Private methods #
    ###################

    def _construct_Rsas_from_Rsp(self, Rsp):
        """
        Constructs reward R[s][a][s'] from a R[s']. (Sometime's it's easier
        to construct a reward function based on state alone).

        Parameters
        ----------
        Rsp : np.array (n_states)
            R[s'] reward function.

        Returns
        -------
        np.ndarray (n_states, n_actions, n_states)
            R[s][a][s'] format of R[s'].

        """
        # Construct R[s][a][s']
        Rsas = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for sp in range(self.n_states):
                    Rsas[s][a][sp] = Rsp[sp]
        return Rsas

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

    # Compute optimal policy
    pi_opt = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        start_idx = s*env.n_actions
        end_idx = s*env.n_actions+env.n_actions
        pi_opt[s] = res.x[start_idx:end_idx].argmax()

    return pi_opt
