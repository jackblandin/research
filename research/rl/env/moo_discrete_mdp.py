import gym
import numpy as np
import pandas as pd
from research.rl.env.helpers import play_n_episodes
from research.rl.env.discrete_mdp import DiscreteMDP, compute_optimal_policy


class MultiObjectiveDiscreteMDP(DiscreteMDP):

    def __init__(self, state_dims, action_dims, obs_dims,
                 max_steps_per_episode, Osaso_dtype='float64', verbose=False,
                 args={}, obj_weights=np.array([1, 0, 0]), T=None, Osaso=None,
                 Rsas=None):
        """
        Additional Parameters beyond DiscreteMDPs
        -----------------------------------------
        obj_weights : np.array<int>
            Objective preference weights (lambdas) for multi-objective a priori
            method.

        Additional Attributes beyond DiscreteMDPs
        -----------------------------------------
        n_objectives : int
            Number of reward objectives.
        Rsasl : np.ndarray (n_states, n_actions, n_states, n_objectives)
            4-dim matrix where index [s][a][s'][l] represents the reward
            component for the l'th objective transitioning from state s to
            state s' after taking action a.  Computed from `_reward_sasl()`
            injected method.
        moo_reward_episode_memory : list<list<np.array<float>>>
            History of rewards observed in each episode broken dow by each
            objective.
        """
        self.obj_weights = np.array(obj_weights)
        self.n_objectives = len(obj_weights)
        self.moo_reward_episode_memory = []

        super().__init__(state_dims, action_dims, obs_dims,
                         max_steps_per_episode, Osaso_dtype,
                         verbose, args, T=T, Osaso=Osaso, Rsas=Rsas)

        # Build multi-objective reward matrix Rsasl from injected method
        larr = np.arange(self.n_objectives)
        self.Rsasl = np.array([
            np.array([
                np.array([
                    self._reward_sasl(s, a, sp, larr) for sp in range(
                        self.n_states)
                    ]) for a in range(self.n_actions)
            ]) for s in range(self.n_states)
        ])

    def step(self, action):
        """
        The agent takes a step in the environment.
        Seperate method needed from DiscreteMDP since the
        moo_reward_episode_memory also needs to be updated.

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
        moo_rewards = self._get_moo_rewards(
            s=prev_state, a=action, sp=new_state)

        # Update current state to new state
        self.cur_state = new_state

        # Compute observation from current state
        ob = self._get_obs(s=prev_state, a=action, sp=new_state)

        # Update action, observation and reward histories
        self.action_episode_memory[self.cur_episode].append(action)
        self.observation_episode_memory[self.cur_episode].append(ob)
        self.reward_episode_memory[self.cur_episode].append(reward)
        self.moo_reward_episode_memory[self.cur_episode].append(moo_rewards)

        # Recompute done since action may have modified it
        done = self.cur_step >= self.max_steps_per_episode

        if self.verbose:
            logging.info('\t' + self.render_state(self.cur_state))

        return ob, reward, done, {}

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Seperate method needed from DiscreteMDP since the
        moo_reward_episode_memory also needs to be updated.

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
        self.moo_reward_episode_memory.append([])

        # Update action, observation and reward histories
        self.action_episode_memory[self.cur_episode].append(-1)
        self.observation_episode_memory[self.cur_episode].append(ob)
        self.reward_episode_memory[self.cur_episode].append(-1)
        self.moo_reward_episode_memory[self.cur_episode].append(
            -1*np.ones(self.n_objectives))

        if self.verbose:
            logging.info(f'Episode {self.cur_episode}')

        if self.verbose:
            logging.info('\t' + self.render_state(self.cur_state))

        return ob

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
        feat_cols = ['z', 'y0', 'y1', 'c', 'yd']
        moo_cols = [f'r{i}' for i in range(self.n_objectives)]
        weighted_moo_cols = [f'r{i}_w' for i in range(self.n_objectives)]
        df = pd.DataFrame([],
                columns=['episode', 'timestep', *feat_cols, 'a', 'r',
                         *moo_cols, *weighted_moo_cols])

        for ep in range(n_eps):
            obss = self.observation_episode_memory[ep+1]
            acts = self.action_episode_memory[ep+1]
            rewards = self.reward_episode_memory[ep+1]
            moo_rewards = np.array(self.moo_reward_episode_memory[ep+1])
            weighted_moo_rewards = np.multiply(moo_rewards, self.obj_weights)
            feats = np.ndarray((len(obss), len(self.state_dims)))
            for i, s in enumerate(obss):
                feats[i] = np.array(self._state_to_feats[s])
            ep_df = pd.DataFrame(feats, columns=feat_cols)
            ep_df['a'] = acts[1:] + [np.nan]  # Actions are offset by 1 step
            ep_df['episode'] = ep
            ep_df['timestep'] = np.arange(n_steps)
            ep_df['r'] = rewards[1:] + [np.nan]  # Rewards are offset by 1 step
            for obj in range(self.n_objectives):
                ep_df[moo_cols[obj]] = np.append(
                    moo_rewards[1:, obj], [np.nan])  # Offset by 1 step
                ep_df[weighted_moo_cols[obj]] = np.append(
                    weighted_moo_rewards[1:, obj], [np.nan])  # Offset by 1 stp
            df = pd.concat([df, ep_df])

        df = df.reset_index(drop=True)

        return df

    def _reward_sas(self, s, a, sp):
        """
        Returns the reward value (and moo reward array) obtained after
        taking action `a` in state `s` and transitioning into new
        state `sp`. Used to construct `Rsas`.

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
        float, np.array<float>
            Reward value and the moo reward array.
        """
        larr = np.arange(self.n_objectives)
        lrs = self._reward_sasl(s, a, sp, larr)
        reward = lrs.dot(self.obj_weights) / self.obj_weights.sum()
        return reward

    def _get_moo_rewards(self, s, a, sp):
        """
        Returns the moo rewards based on s, a, s'.

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
        np.array<float>
            Moo rewards.
        """
        return self.Rsasl[s,a,sp]
