import numpy as np

from .feature_transformer import ObservationAsStatesTransformer


class QLearnerObsSingle:

    def __init__(self, env, initial_alpha=.1, gamma=.9, alpha_decay=0,
                 feature_transformer=None, num_states=None, num_actions=None):
        """
        Simple Q Learner with just observations as states. The action is
        associated with the last observation.

        Parameters
        ----------
        env : gym.Env
            OpenAI Gym environment
        initial_alpha : float
            Learning rate.
        gamma : float
            Discount factor.
        alpha_decay : float, default 0
            Learning rate alpha will decay at 1/_n_updates**_alpha_decay.
        feature_transformer : object, default None
            Object that transforms observations into states.
        num_states : int
            Number of states in env observation space.
        num_actions : int
            Number of actions in env action space.

        Attributes
        ----------
        feature_transformer : object
            Transforms raw state into representation, usually a reduced one.
        Q : 2D numpy array
            Q <state,value> matrix where
                - axis0 index is transformed observation
                - axis1 index is action
                - value is Q value.
            E.g. Q[1][2] represents the Q value for taking action 2 in
            (transformed) state 1.
        _n_updates : int
            Number of updates made to Q matrix.
        """
        self.env = env
        self.initial_alpha = initial_alpha
        self.gamma = gamma
        self._alpha_decay = alpha_decay
        if feature_transformer is None:
            self.feature_transformer = ObservationAsStatesTransformer(env)
        else:
            self.feature_transformer = feature_transformer
        if num_states is None:
            num_states = env.observation_space.n
        if num_actions is None:
            num_actions = env.action_space.n
        self.Q = np.random.uniform(low=0, high=0,
                                   size=(num_states, num_actions))
        self._n_updates = 0

    def predict(self, o):
        """
        Parameters
        ----------
        o : list or array-like
            A single observation.

        Returns
        -------
        numpy array
            Each index is an action, and values are Q values of taking that
            action, given the transformed state.
        """
        o_trans = self.feature_transformer.transform(o)
        return self.Q[o_trans]

    def update(self, otm1, atm1, r, ot, at):
        """
        Performs TD(0) update on the model.

        Parameters
        ----------
        otm1 : list or array-like
            Previous observation (o t "minus" 1)
        atm1 : int
            Previous action
        r : float
            Reward of taking previous action given previous observation
        ot : list or array-like
            Current observation.
        at : int
            Action chosen to update the model. Usually this is the best action.

        Returns
        -------
        None
        """
        otm1_trans = self.feature_transformer.transform(otm1)
        ot_trans = self.feature_transformer.transform(ot)
        G = r + self.gamma*self.Q[ot_trans, at]
        alpha = self.initial_alpha / (self._n_updates+1)**self._alpha_decay
        self.Q[otm1_trans, atm1] += alpha*(G - self.Q[otm1_trans, atm1])
        self._n_updates += 1

    def sample_action(self, o, eps):
        """
        Uses epsilon greedy strategy to either sample a random action, or
        select the best action.

        Parameters
        ----------
        o : list or array-like
            A single observation.
        eps : float
            Epsilon used in epsilon-greedy.

        Returns
        -------
        int
            An action.
        """
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return self.best_action(o)

    def best_action(self, o):
        """
        Returns the action with the highest q value, given an observation.

        Parameters
        ----------
        o : object
            Observation.

        Returns
        -------
        int
            Best action given observation.
        """
        return np.argmax(self.predict(o))

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
            zero_qvals = self.Q[0].round(2)
            one_qvals = self.Q[1].round(2)
            s += '\n{: >6} \tACTION 0 | ACTION 1'.format('')
            s += '\n\t-------- | --------'
            s += '\nOBS 0 {:>10} | {:>8}'.format(*zero_qvals)
            s += '\nOBS 1 {:>10} | {:>8}'.format(*one_qvals)
        elif env_class == 'TigerEnv':
            # st_qvals = self.Q[0].round(2)
            gl_qvals = self.Q[1].round(2)
            gr_qvals = self.Q[2].round(2)
            s += '\n{: >10} \tOPEN LEFT | OPEN RIGHT | LISTEN'.format('')
            s += '\n\t\t--------- | ---------- | ------'
            # s += '\nSTART: {: >18} | {: >10} | {: >6}'.format(*st_qvals)
            s += '\nGROWL LEFT {: >14} | {: >10} | {: >6}'.format(*gl_qvals)
            s += '\nGROWL RIGHT: {: >12} | {: >10} | {: >6}'.format(*gr_qvals)
        else:
            raise ValueError('Don\'t know how to represent model for this env')

        s += '\n'
        return s
