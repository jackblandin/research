"""
Started 06/17/2019

Objectives
----------
2. Verify MLPRegressor works in DQN. Use single observation as input.

Notes
-----
"""

import numpy as np
import gym
import matplotlib.pyplot as plt
from research.neural_networks.mlp import MLPRegressor, ReLU
from copy import deepcopy


class NumpyDQNObsSingle:
    """A Deep Q Network. Uses MLPRegressor as deep Q network.

    Uses a single observation as input to the Q network.

    Parameters
    ----------
    env : gym.Env
        OpenAI Gym environment
    feature_transformer : object
        Transforms observation into a the Q value space. Must have
        transform(obs)` instance method.
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
    """

    def __init__(self, env, feature_transformer, D, K, hidden_layer_opts,
                 gamma, start_obs, max_experiences=10000, min_experiences=5):
        self.env = env
        self.feature_transformer = feature_transformer
        self.D = D
        self.K = K
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

    def predict(self, obs, use_target=False):
        """
        Returns an array where index is action and value is Q value for taking
        the action.

        Parameters
        ----------
        obs : array-like, shape (D)
            Input observation.
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
        X = np.array(obs).reshape(1, -1)

        X = self.feature_transformer.transform(X)

        action_values = np.zeros((len(self.tqnets)))
        qnet_Zs = [None for tqnet in self.tqnets]

        if use_target:
            for i, tqnet in enumerate(self.tqnets):
                action_values[i], qnet_Zs[i] = tqnet.forward(X)
        else:
            for i, mqnet in enumerate(self.mqnets):
                action_values[i], qnet_Zs[i] = mqnet.forward(X)

        return action_values.reshape(1, -1), qnet_Zs

    def train(self, n):
        """
        Randomly selects a batch (of 1) from experience replay and performs an
        iteration of gradient descent on the main Q network (hidden layers).

        Parameters
        ----------
        n : int
            Timestep.

        Returns
        -------
        None
        """
        ##
        # Return early if we don't have enough experiences.
        ##
        num_exp = len(self.experience['s'])
        if num_exp < self.min_experiences:
            print('too few experiences', num_exp)
            return

        # Randomly select a batch of 1 (not sequence) of observations from
        # replay buffer.
        idx = np.random.choice(num_exp, size=1, replace=False)[0]
        state = self.experience['s'][idx]
        action = self.experience['a'][idx]
        reward = self.experience['r'][idx]
        next_state = self.experience['s2'][idx]
        done = self.experience['done'][idx]

        ##
        # Compute predicted Q values using main network. These are the
        # "predictions".
        ##
        action_values, qnet_Zs = self.predict(state)
        # Note that best action is used to get Q value, but is not used when
        # updating the model via backprop.
        Q = action_values[:, action]
        Z = qnet_Zs[action]

        ##
        # Using the target network, compute the expected Q values after taking
        # each possible action, given the s2 observation from the replay
        # buffer. These values will used as the "target" values since Q
        # learning treats G=r+gamma*Q[n+1] as the target for Q[n].
        ##
        next_action_values, _ = self.predict(next_state, use_target=True)
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
        G = np.atleast_2d(G)
        Q = np.atleast_2d(Q)
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
        ##
        # Compute predicted outputs and accumulate hidden layer outputs. We
        # accumulate the hidden layer outputs so that we can run backprop.
        ##
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            action_values, _ = self.predict(obs)
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
        """
        String representation of the model.

        Uses `env` attribute to know observations and actions to show.

        Returns
        -------
        str
            A string showing the Q values of each state/action.
        """
        env_class = self.env.__class__.__name__

        s = '\n'
        if env_class == 'GreaterThanZeroEnv':
            neg_one = self.predict([-1])[0].round(3)
            neg_half = self.predict([-.5])[0].round(3)
            zero = self.predict([0])[0].round(3)
            pos_half = self.predict([.5])[0].round(3)
            pos_one = self.predict([1])[0].round(3)
            s += '\n\t   ACTION 0 | ACTION 1'.format('')
            s += '\n\t   -------- | --------'
            s += '\nOBS -1.0 {:>10} | {:>8}'.format(*neg_one[0])
            s += '\nOBS -0.5 {:>10} | {:>8}'.format(*neg_half[0])
            s += '\nOBS  0.0 {:>10} | {:>8}'.format(*zero[0])
            s += '\nOBS  0.5 {:>10} | {:>8}'.format(*pos_half[0])
            s += '\nOBS  1.0 {:>10} | {:>8}'.format(*pos_one[0])
        elif env_class == 'TigerEnv':
            gl_qvals, _ = self.predict([0])
            gr_qvals, _ = self.predict([1])
            st_qvals, _ = self.predict([2])
            gl_qvals = gl_qvals.round(3)
            gr_qvals = gr_qvals.round(3)
            st_qvals = st_qvals.round(3)
            s += '\n{: >10} \tOPEN LEFT | OPEN RIGHT | LISTEN'.format('')
            s += '\n\t\t--------- | ---------- | ------'
            s += '\nGROWL LEFT {: >14} | {: >10} | {: >6}'.format(*gl_qvals)
            s += '\nGROWL RIGHT: {: >12} | {: >10} | {: >6}'.format(*gr_qvals)
            s += '\nSTART: {: >18} | {: >10} | {: >6}'.format(*st_qvals)
        elif env_class == 'NotXOREnv':
            s = self.env.q_values(self)
            return s
        else:
            raise ValueError('Don\'t know how to represent model for this env')

        s += '\n'
        return s


def play_one(env, model, eps, gamma, copy_period):
    """
    Plays a single episode. During play, the model is updated, and the total
    reward is accumulated and returned.

    Parameters
    ----------
    env : gym.Env
        Environment.
    model : NumpyDQNObsSingle
        Model instance.
    eps : numeric
        Epsilon used in epsilon-greedy.
    gamma : numeric
        Discount factor.
    copy_period : int
        The number of steps b/w each copy of main Q network to target Q
        network.

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

        # Choose an action based on current observation.
        action = model.select_action(obs, eps)
        prev_obs = obs

        # Take chosen action.
        obs, reward, done, _ = env.step(action)

        totalreward += reward

        # Update the model
        model.add_experience(prev_obs, action, reward, obs, done)
        model.train(iters)

        iters += 1

        # If at copy period, copy the main model to the target model
        if iters % copy_period == 0:
            model.copy_mtqnets()

    return totalreward


def main(copy_period=50, hidden_layer_sizes=[10, 10], N=500):
    env = gym.make('Tiger-v0')
    gamma = 0.99
    start_obs = env.reset()

    # Define D - the number of components in the observations space
    D = len(env.observation_space.sample())
    # Define K - the number of possible actions
    K = env.action_space.n
    # Define the hidden layers of the model
    hidden_layer_opts = {'hidden_layer_sizes': hidden_layer_sizes, 'Z': ReLU()}
    # Define model
    model = NumpyDQNObsSingle(D, K, hidden_layer_opts, gamma, start_obs)

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


def running_avg(totalrewards, t):
    return totalrewards[max(0, t-100):(t+1)].mean()


def plot_running_avg(totalrewards):
    N = len(totalrewards)
    ravg = np.empty(N)
    for t in range(N):
        ravg[t] = running_avg(totalrewards, t)
    plt.plot(ravg)
    plt.title('Running Average')
    plt.show()


if __name__ == '__main__':
    main()
