"""
Started 05/15/2019

Objectives
----------
1   Learn the fundamentals of DQNs.
2   Learn how DQNs converge on the value function relative to previous naive
    strategies.
3   Learn basics of tensorflow through implementation.

Notes
-----
TensorFlow Operations
    * Nodes that perform computations on or with Tensor objects.
    * After computation, they return zero or more tensors.
    * The returned tensors can be used by other Ops later in the graph.
    * https://stackoverflow.com/questions/43290373/what-is-tensorflow-op-does
tf.reduce_sum
    * Computes the sum of elements across dimensions of a tensor.
Tensorflow variable assignment
    * The statement `x.assign(1)` does not actually assign the value 1 to x.
    * Rather, it creates a tf.Operation that you have to explicitly run in
      order to update the variable.
    * Operation.run() or Session.run() can be used to run the operation.
"""

import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt


class HiddenLayer:
    """
    A version of HiddenLayer that keeps track of params.

    Parameters
    ----------
    name : str
        Unique identifier, used for TensorBoard scope name.
    M1 : int
        Input layer dimension
    M2 : int
        Output layer dimension
    f : function, default tf.nn.tanh
        Activation function
    use_bias : bool, default True
        Whether to use a bias or not. If True, a 'b' attribute will be assigned
        to this class instance, with shape M2 and all values as zero.
    use_name_scope : bool, default False
        Whether to assign weights using the name scope. Set to True when you
        want to view the output in TensorBoard, but be careful there are no
        name collisions between the main and target graphs.
    """

    def __init__(self, name, M1, M2, f=tf.nn.tanh, use_bias=True,
                 use_name_scope=False):
        self.name = name

        if use_name_scope:
            with tf.name_scope(name):
                with tf.name_scope('weights'):
                    self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
                    _variable_summaries(self.W)
                self.params = [self.W]
                self.use_bias = use_bias
                if self.use_bias:
                    with tf.name_scope('biases'):
                        self.b = tf.Variable(np.zeros(M2).astype(np.float32))
                        _variable_summaries(self.b)
                    self.params.append(self.b)
                self.f = f
        else:
            self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
            self.params = [self.W]
            self.use_bias = use_bias
            if self.use_bias:
                self.b = tf.Variable(np.zeros(M2).astype(np.float32))
                self.params.append(self.b)
            self.f = f

    def forward(self, X):
        """
        Runs one forward pass of NN.

        Parameters
        ----------
        X : The output of a hidden layer

        Returns
        -------
        numeric?
            The activation value (I think a number b/w 0 and 1)

        """
        if self.use_bias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        return self.f(a)


class DQN:
    """
    A Deep Q Network. Uses TensorFlow. Only most recent observation is used,
    therefore, each set of observations must be MDP.

    Parameters
    ----------
    name : str
        Unique identifier. Used to delineate variables when there are multiple
        graphs.
    session : tf.InteractiveSession
        TensorFlow session.
    D : int
        Number of components in the observation space.
    K : int
        Number of components in the action space.
    hidden_layer_sizes : array-like
        Array of integers where each integer is the number of nodes in the
        corresponding hidden layer. Note that the number of hidden layers is
        then defined by the length of this array.
    gamma : numeric
        Number between zero and one. Discount factor.
    max_experiences : int
        Maximum number of replay tuples to include in experience replay.
    min_experiences : int
        Minimum number of replay tuples required to start training training the
        Q network.
    batch_size : int, default 32
        Number of observations to include in batch updates.
    log_summaries : bool, default False
        Whether to log summaries for TensorBoard.
    logs_dir: str, default None
        Filepath of the log directory for writing TensorBoard output. Required
        if log_summaries is True.

    Attributes
    ----------
    layers : list
        A list of HiddenLayer instances. This represents the main Q network.
    params : list
        A list of params for each HiddenLayer. This is used when copying the
        main Q network to the target network.
    X : tf.placeholder
        Tensor placeholder for the observation inputs?
    G : tf.placeholder
        Tensor placeholder for the Q values?
    actions : tensor placeholder
        Tensor placeholder for actions.
    predict_op : tf?
        The result of calling layer.forward() on each the layers.
    cost : The SSE of G - Q
        Subtrac
    train_op : tf.train.AdamOptimizer
        The tf operator for performing gradient descent on the main Q network
        (hidden layers). The composition of the operator is as follows:
            train_op                        - AdamOptimizer.minimize(cost)
                cost                        - SSE(G - selected_action_values)
                    G                       - placeholder
                    selected_action_values  - sum(Y_hat * onehot(actions))
                        actions             - placeholder
                        Y_hat               - output of final layer.forward()
                            X               - input
                            layers          - list of HiddenLayers.
    experience : dict
        Experience replay. Keys are 's', 'a', 'r', 's2', 'done', and values
        are lists.
    merged_summaries_ : ?
        Result of running tf.summary.merge_all().
    train_tb_writer_ : tf.summary.FileWriter
        Used for writing TensorBoard training summaries.
    cost_tb_writer_ : tf.summary.FileWriter
        Used for writing TensorBoard cost summaries.
    """

    def __init__(self, name, session, D, K, hidden_layer_sizes, gamma,
                 max_experiences=10000, min_experiences=100, batch_size=32,
                 log_summaries=False, logs_dir=None):
        self.name = name
        self.session = session
        self.K = K
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_size = batch_size
        self.gamma = gamma
        self.log_summaries = log_summaries
        self.logs_dir = logs_dir

        ##
        # Create the hidden layers of the main graph.
        ##
        self.layers = []
        M1 = D
        for idx, M2 in enumerate(hidden_layer_sizes):
            name = '{}-H{}'.format(self.name, idx)  # Used for TensorBoard
            layer = HiddenLayer(name, M1, M2,
                                use_name_scope=self.log_summaries)
            self.layers.append(layer)
            M1 = M2

        ##
        # Create the final layer of the main graph.
        ##
        layer = HiddenLayer('{}-Output'.format(self.name), M1, K, lambda x: x,
                            use_name_scope=self.log_summaries)
        self.layers.append(layer)

        ##
        # Collect params for copying the main graph to the target graph.
        ##
        self.params = []
        for layer in self.layers:
            self.params += layer.params

        ##
        # Define the tensorflow placehoders for the inputs, targets, and
        # actions.
        ##
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

        ##
        # Define the forward() calls b/w each hidden layer, and the final
        # output.
        ##
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = Z
        self.predict_op = Y_hat

        ##
        # Define the actual of the output of the Q network, which will be the
        # output of the NN associated with the action that was taken. Remember
        # that there is one NN for each action.
        ##
        selected_action_values = tf.reduce_sum(
            Y_hat * tf.one_hot(self.actions, K),
            reduction_indices=[1])

        ##
        # Define the optimizer for the network.
        ##
        if self.log_summaries:
            with tf.name_scope('cost'):
                self.cost = tf.reduce_sum(
                    tf.square(self.G-selected_action_values))
            with tf.name_scope('train'):
                self.train_op = tf.train.AdamOptimizer(1e-3).minimize(
                    self.cost)
        else:
            self.cost = tf.reduce_sum(tf.square(self.G-selected_action_values))
            self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.cost)

        ##
        # Merge all the TensorBoard summaries and write them out to ./logs
        ##
        if self.log_summaries:
            self.merged_summaries_ = tf.summary.merge_all()
            self.train_tb_writer_ = tf.summary.FileWriter(
                self.logs_dir+'/train', self.session.graph)
            self.cost_tb_writer_ = tf.summary.FileWriter(self.logs_dir+'/cost')

        ##
        # Create the replay memory.
        ##
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}

    def copy_from(self, other):
        """
        Copies tensorflow parameters of some (usually target) DQN network to
        THIS DQN network.

        Parameters
        ----------
        other : DQN
          DQN instance to copy parameters from.

        Returns
        -------
        None
        """
        ##
        # Collect all the operators.
        ##
        ops = []
        my_params = self.params
        other_params = other.params
        for p, q in zip(my_params, other_params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)

        ##
        # Run all the operations
        ##
        self.session.run(ops)

    def predict(self, X):
        """
        Returns an array where index is action and value is Q value for taking
        the action.

        Parameters
        ----------
        X : array-like
            Raw observation input (untransformed state). NOTE: if X is more
            than 1-dimensional, the inputs are INDEPENDENT. I.e. the inputs are
            NOT a sequence of observations - they will receive separate and
            independent predictions. This is useful for batch purposes.

        Returns
        -------
        array-like?
            Action-value array.
        """
        ##
        # Run the predict operator, feeding in the input X.
        ##
        # X = np.atleast_2d(X).flatten().reshape(1, -1)
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def train(self, target_network, n):
        """
        Randomly selects a batch from experience replay and performs an
        iteration of gradient descent on the main Q network (hidden layers).

        Parameters
        ----------
        target_network : DQN
            The target Q network.
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
            return

        ##
        # Randomly select a batch (not sequence) of observations from replay buffer.
        ##
        idx = np.random.choice(num_exp, size=self.batch_size, replace=False)
        states = [self.experience['s'][i] for i in idx]
        actions = [self.experience['a'][i] for i in idx]
        rewards = [self.experience['r'][i] for i in idx]
        next_states = [self.experience['s2'][i] for i in idx]
        dones = [self.experience['done'][i] for i in idx]

        ##
        # Using the target network, compute the expected Q values after taking
        # each possible action, given the s2 observation sequence from the
        # replay buffer.
        ##
        action_values = target_network.predict(next_states)

        ##
        # Compute the resulting Q network after taking the previously computed
        # action.
        ##
        next_Q = np.max(action_values, axis=1)

        ##
        # Compute the updated G values using the rewards and next_Q.
        ##
        # TODO try doing this in a list and see if that messes this up
        # NOTE: The "targets" here are actually "G"
        targets = [r + self.gamma * nQ if not done else r
                   for r, nQ, done in zip(rewards, next_Q, dones)]

        ##
        # Every tenth train step, record the cost summary.
        ##
        if self.log_summaries and n % 10 == 0:
            summary, _ = self.session.run(
                [self.merged_summaries_, self.cost], feed_dict={
                    self.X: states,
                    self.G: targets,
                    self.actions: actions})
            self.cost_tb_writer_.add_summary(summary, n)

        ##
        # Update the actual network by running the optimizer.
        ##
        if self.log_summaries:
            summary, _ = self.session.run([self.merged_summaries_,
                                           self.train_op],
                                          feed_dict={self.X: states,
                                                     self.G: targets,
                                                     self.actions: actions})
        else:
            self.session.run([self.train_op],
                             feed_dict={self.X: states,
                                        self.G: targets,
                                        self.actions: actions})

        if self.log_summaries:
            self.train_tb_writer_.add_summary(summary, n)

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

    def sample_action(self, x, eps):
        """
        Returns an action. Uses epsilon greedy for trading off b/w exploration
        and explotaition.

        Parameters
        ----------
        x : object
            An observation.
        eps : numeric
            Epsilon used for epsilon-greedy.

        Returns
        -------
        int
            A single action.
        """
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            X = np.atleast_2d(x)
            return np.argmax(self.predict(X)[0])


def play_one(env, model, tmodel, eps, gamma, copy_period, max_steps=10):
    """
    Plays a single episode. During play, the model is updated, and the total
    reward is accumulated and returned.

    Parameters
    ----------
    env : gym.Env
        Environment.
    model : DQN
        Main Q network.
    tmodel : DQN
        Target Q network.
    eps : numeric
        Epsilon used in epsilon-greedy.
    gamma : numeric
        Discount factor.
    copy_period : int
        The number of steps b/w each copy of main Q network to target Q
        network.
    max_steps : int, default 10
        Max number of steps allowed in a single episode.

    Returns
    -------
    numeric
        Total reward accumualted during episode.

    """
    obs = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done and iters < max_steps:
        ##
        # Choose an action based on current observation.
        ##
        action = model.sample_action(obs, eps)
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
        model.train(tmodel, iters)

        iters += 1

        ##
        # If at copy period, copy the main model to the target model
        ##
        if iters % copy_period == 0:
            tmodel.copy_from(model)

    return totalreward


def main(logs_dir, copy_period=50, hidden_layer_sizes=[10, 10], N=500,
         batch_size=32):
    env = gym.make('CartPole-v0')
    gamma = 0.99

    # Define D - the number of components in the observations space
    D = len(env.observation_space.sample())
    # Define K - the number of possible actions
    K = env.action_space.n
    # Define session - The tensorflow interactive session
    session = tf.InteractiveSession()
    # Define model - the main DQN model
    model = DQN('main', session, D, K, hidden_layer_sizes, gamma, session,
                batch_size=batch_size, logs_dir=logs_dir)
    # Define tmodel - the target DQN model
    tmodel = DQN('target', session, D, K, hidden_layer_sizes, gamma, session,
                 batch_size=batch_size)
    # Define init - the tensorflow global variables initializer
    init = tf.global_variables_initializer()
    # Run the session while passing in the global variables initializer
    session.run(init)

    totalrewards = np.zeros(N)
    for n in range(N):
        eps = 1.0/np.sqrt(n+1)
        totalreward = play_one(env, model, tmodel, eps, gamma, copy_period,
                               logs_dir)
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


def _variable_summaries(var):
    """
    Helper method for TensorBoard visualization.

    Parameters
    ----------
    var : tf.Variable
        The tensorflow variable for the node you're attaching summary info to.

    Returns
    -------
    None
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


if __name__ == '__main__':
    main()

