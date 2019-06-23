"""
Started 06/14/2019

Objectives
----------
1. Use MLPRegressor as deep Q network. This is a class I wrote using numpy.

Notes
-----
"""
import numpy as np
import gym
import matplotlib.pyplot as plt
from copy import deepcopy

from research.neural_networks.mlp import MLPRegressor, ReLU
from ....utils.tiger_env_utils import env_translate_obs, env_translate_action


class NumpySeqDQN:
    """A Deep Q Network. Uses MLPRegressor as deep Q network.

    Uses a sequence of observations as input to the Q network.

    Parameters
    ----------
    env : gym.Env
        OpenAI Gym environment.
    D : int
        Number of components in the observation space.
    K : int
        Number of components in the action space.
    hidden_layer_opts : dict
        Hidden layer options. Keys are:
            hidden_layer_sizes : array-like
                Array of integers where each integer is the number of nodes in
                the corresponding hidden layer. Note that the number of hidden
                layers is then defined by the length of this array.
            Z : research.neural_networks.mlp.Activation
                Activation function.
            learning_rate : numeric, optional
                Neural network Learning rate.
            reg : numeric, optional
                Neural network regularization parameter.
            mu :numeric, optional
                Momentum parameter.
            clip_thresh : numeric
                Max allowed gradient, before it's clipped.
    gamma : numeric
        Number between zero and one. Discount factor.
    obs_seq_len : int
        Number of most recent observations to pass into the Q network.
    start_obs : object
        Starting observation. This will be duplciated and filled in as the
        `last_n_obs` when there aren't enough observations yet.
    max_experiences : int, default 1000
        Maximum number of replay tuples to include in experience replay.
    min_experiences : int, 5
        Minimum number of replay tuples required to start training training the
        Q network.

    Attributes
    ----------
    mqnets : list<research.neural_networks.mlp.MLPRegressor>
       Main deep Q networks. One per action.
    tqnets : list<research.neural_networks.mlp.MLPRegressor>
       Target deep Q networks. One per action.
    experience : dict
        Experience replay. Keys are 's', 'a', 'r', 's2', 'done', and values
        are lists.
    last_n_obs : list
        A sequence of the most recent <obs_seq_len> observations.
    train_obs_seq_counts : dict<str, int>
        Dict with stringified observation sequences as keys and values as the
        number of times this sequence was predicted on. The observations and
        actions are referring to the sets that are sampled from experience
        replay and used to update the DQN. Currently this is only used for
        debuggging.
    train_obs_seq_action_counts : dict<str, int>
        Dict with stringified observation sequence + action as keys and values
        as the count of taking that action on that observation sequence. The
        observations and actions are referring to the sets that are sampled
        from experience replay and used to update the DQN. Currently this is
        only used for debugging.
    """

    def __init__(self, env, D, K, hidden_layer_opts, gamma, obs_seq_len,
                 start_obs, max_experiences=10000, min_experiences=5):
        self.env = env
        self.K = K
        self.obs_seq_len = obs_seq_len
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.gamma = gamma

        self.mqnets = []
        self.tqnets = []

        for i in range(self.K):
            mqnet = MLPRegressor(D, K=1, **hidden_layer_opts)
            tqnet = deepcopy(mqnet)
            self.mqnets.append(mqnet)
            self.tqnets.append(tqnet)

        # Create the replay memory.
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}

        self.last_n_obs = [start_obs for i in range(self.obs_seq_len)]
        self.train_obs_seq_counts = {}
        self.train_obs_seq_action_counts = {}

    def predict(self, obs_seq, use_target=False):
        """Returns array of Q values where index is action and value is Q value
        for taking the action.

        This method is distinct from `_predict` in that
        in that it doesn't return the Hidden layer outputs. This method exists
        so that it conforms to the model.predict schemas of other model types.

        Parameters
        ----------
        obs_seq : array-like, shape (D)
            Input observation sequence.
        use_target : bool, default False
            If True, uses the target Q network to make predictions.

        Returns
        -------
        list<numeric>, shape (K)
            Action-value array.
        """
        preds = self._predict(obs_seq, use_target)[0]
        return preds

    def _predict(self, obs_seq, use_target=False):
        """Returns an array where index is action and value is Q value for
        taking the action.

        Parameters
        ----------
        obs_seq : array-like, shape (D)
            Input observation sequence.
        use_target : bool, default False
            If True, uses the target Q network to make predictions.

        Returns
        -------
        tuple, shape(2)
            list<numeric>, shape (K)
                Action-value array.
            list<np.ndarray>, shape (K)
                Hidden layer outputs for ALL qnets.
        """
        # Convert list of lists to array of arrays
        X = np.array([np.array(obs) for obs in obs_seq])
        # Flatten the sequences into single array
        X = X.flatten().reshape(1, -1)

        action_values = np.zeros((len(self.tqnets)))
        qnet_Zs = [None for tqnet in self.tqnets]

        if use_target:
            for i, tqnet in enumerate(self.tqnets):
                action_values[i], qnet_Zs[i] = tqnet.forward(X)
        else:
            for i, mqnet in enumerate(self.mqnets):
                action_values[i], qnet_Zs[i] = mqnet.forward(X)

        return action_values.reshape(1, -1), qnet_Zs

    def train(self, n, store_seq_counts=True):
        """
        Randomly selects a batch from experience replay and performs an
        iteration of gradient descent on the main Q network (hidden layers).

        Parameters
        ----------
        n : int
            Timestep.
        store_seq_counts : bool, default True
            Whether to store obs. sequence and action counts.

        Returns
        -------
        None
        """
        seqlen = self.obs_seq_len  # Rename for convenience
        ##
        # Return early if we don't have enough experiences.
        ##
        num_exp = len(self.experience['s'])
        if num_exp < self.min_experiences:
            print('too few experiences', num_exp)
            return

        ##
        # Randomly select a sequence of observations from replay buffer.
        ##
        # Check that no "dones" are in the middle of the sequence.
        invalid_seq = True
        while invalid_seq:
            # Start of sequence timestep
            nS = np.random.choice(num_exp - seqlen - 1)
            states = np.array(self.experience['s'][nS:nS+seqlen])
            action = self.experience['a'][nS+seqlen-1]
            reward = self.experience['r'][nS+seqlen-1]
            next_states = self.experience['s2'][nS:nS+seqlen]
            dones = self.experience['done'][nS:nS+1+seqlen]
            invalid_seq = any(dones[0:-1])

        done = dones[-1]

        if store_seq_counts:
            ##
            # Update training observation sequence counts
            ##
            transl_obs = [env_translate_obs(o) for o in states]
            transl_obs = ', '.join(transl_obs)
            if transl_obs in self.train_obs_seq_counts:
                self.train_obs_seq_counts[transl_obs] += 1
            else:
                self.train_obs_seq_counts[transl_obs] = 1

            ##
            # Update training observation sequence + action counts
            ##
            transl_act = env_translate_action(action)
            obs_seq_act = ' => '.join([transl_obs, transl_act])
            if obs_seq_act in self.train_obs_seq_action_counts:
                self.train_obs_seq_action_counts[obs_seq_act] += 1
            else:
                self.train_obs_seq_action_counts[obs_seq_act] = 1

        ##
        # Compute predicted Q values using main network. These are the
        # "predictions".
        ##
        action_values, qnet_Zs = self._predict(states)
        Q = action_values[:, action]
        Z = qnet_Zs[action]

        ##
        # Using the target network, compute the expected Q values after taking
        # each possible action, given the s2 observation sequence from the
        # replay buffer. These values will used as the "target" values since
        # Q learning treats G=r+gamma*Q[n+1] as the target for Q[n].
        ##
        next_action_values, _ = self._predict(next_states, use_target=True)
        # Uses the best action, not the action taken.
        next_Q = np.max(next_action_values, axis=1)

        ##
        # Compute the updated G values using the rewards and next_Q.
        ##
        if not done:
            G = reward + self.gamma * next_Q
        else:
            G = reward

        ##
        # Update the main network. Only need to update the nn for the action
        # that was taken.
        ##
        Q = np.atleast_2d(Q)
        G = np.atleast_2d(G)

        self.mqnets[action].update(G, Q, Z)

    def add_experience(self, s, a, r, s2, done):
        """
        Adds an experience replay tuple to the replay buffer. If the buffer is
        full, then a tuple is popped from the beginning of the queue.

        Parameters
        ----------
        s : object
            Previous observation.
        a : int
            Action taken.
        r : numeric
            Reward received for taking action a after observing observation o.
        s2 : object
            Observation observed after taking action a after receiving
            observation o.
        done : boolean
            Whether the episode ended after taking action a.

        Returns
        -------
        None
        """
        if len(self.experience['s']) >= self.max_experiences:
            for p in ['s', 'a', 'r', 's2', 'done']:
                self.experience[p].pop(0)
        for p, v in zip(['s', 'a', 'r', 's2', 'done'], [s, a, r, s2, done]):
            self.experience[p].append(v)

    def select_action(self, obs, eps):
        """
        Returns an action. Uses epsilon greedy for trading off b/w exploration
        and explotaition. Updates model.last_n_obs with obs.

        Parameters
        ----------
        obs : array-like, shape(D)
            Single observation.
        eps : numeric
            Epsilon used for epsilon-greedy.

        Returns
        -------
        int
            A single action.
        """
        # TODO 05/25/2019 - The sequence of last_n_obs could have observations
        # from the previous episode. How should this be handled?

        # Fetch last_n_obs and prepend to input observation.
        self.last_n_obs.pop(0)
        # Update last_n_obs
        self.last_n_obs.append(obs)

        ##
        # Compute predicted outputs and accumulate hidden layer outputs. We
        # accumulate the hidden layer outputs so that we can run backprop.
        ##
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            action_values, _ = self._predict(self.last_n_obs)
            action = np.argmax(action_values)
            return action

    def copy_mtqnets(self):
        """Copies main Q nets to target Q nets.

        Returns
        -------
        None
        """
        for i, mqnet in enumerate(self.mqnets):
            self.tqnets[i] = deepcopy(mqnet)

    def __str__(self):
        """String representation of the model.

        Returns
        -------
        str
            A string showing the Q values of each state/action.

        Examples
        --------
        >>> model = NumpySeqDQN(D, K, hidden_layer_opts, gamma, obs_seq_len
                                start_obs)
        >>> print(model)
        Obs. Seq      Action 0    Action 1
        ----------  ----------  ----------
        0,0            14.4307     14.1678
        1,1            14.3387     14.2174
        0,1            14.502      14.2673
        1,0            14.323      14.0412
        """
        s = self.env.q_values(self)
        return s


def play_one(env, model, eps, gamma, copy_period, store_seq_counts=True):
    """
    Plays a single episode. During play, the model is updated, and the total
    reward is accumulated and returned.

    Parameters
    ----------
    env : gym.Env
        Environment.
    model : NumpySeqDQN
        Model instance.
    eps : numeric
        Epsilon used in epsilon-greedy.
    gamma : numeric
        Discount factor.
    copy_period : int
        The number of steps b/w each copy of main Q network to target Q
        network.
    store_seq_counts : bool, default True
        Whether to store obs. sequence and action counts.

    Returns
    -------
    numeric
        Total reward accumualted during episode.

    """
    obs = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done:
        ##
        # Choose an action based on current observation.
        ##
        action = model.select_action(obs, eps)
        prev_obs = obs

        ##
        # Take chosen action.
        ##
        obs, reward, done, _ = env.step(action)

        totalreward += reward

        ##
        # Update the model
        ##
        model.add_experience(prev_obs, action, reward, obs, done)
        model.train(iters, store_seq_counts=store_seq_counts)

        iters += 1

        ##
        # If at copy period, copy the main model to the target model
        ##
        if iters % copy_period == 0:
            model.copy_mtqnets()

    return totalreward


def main(copy_period=50, obs_seq_len=2, hidden_layer_sizes=[10, 10], N=500):
    env = gym.make('Tiger-v0')
    gamma = 0.99
    start_obs = env.reset()

    # Define D - the number of components in the observations space
    D = len(env.observation_space.sample())
    # Define K - the number of possible actions
    K = env.action_space.n
    # Define the hidden layers of the model
    hidden_layer_opts = {'hidden_layer_sizes': hidden_layer_sizes,
                         'Z': ReLU()}
    # Define model
    model = NumpySeqDQN(D, K, hidden_layer_opts, gamma, obs_seq_len, start_obs)

    totalrewards = np.zeros(N)
    for n in range(N):
        eps = 1.0/np.sqrt(n+1)
        totalreward = play_one(env, model, eps, gamma, copy_period)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            ravg = running_avg(totalrewards, n)
            print('episode:', n,
                  'total reward:', totalreward,
                  'eps:', eps,
                  'avg reward (last 100):', ravg)

    print('avg reward for last 100 episodes:', totalrewards[-100:].mean())
    print('total steps:', totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)


def running_avg(totalrewards, t, window):
    return totalrewards[max(0, t-window):(t+1)].mean()


def plot_running_avg(totalrewards, window):
    N = len(totalrewards)
    ravg = np.empty(N)
    for t in range(N):
        ravg[t] = running_avg(totalrewards, t, window)
    plt.plot(ravg)
    plt.title('Running Average')
    plt.show()


if __name__ == '__main__':
    main()
