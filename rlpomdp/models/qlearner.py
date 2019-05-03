# 3rd party
import numpy as np

# Local
from ..feature_transformers.simple import TigerFeatureTransformer


class QLearner:
    def __init__(self, env, alpha=.1, gamma=.9):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.feature_transformer = TigerFeatureTransformer(env)
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        # axis0 is transformed observation, axis1 is action, value is Q value
        self.Q = np.random.uniform(low=-1, high=1,
                                   size=(num_states, num_actions))

    def predict(self, o):
        """
        Returns an array where index is an action, and values are the Q values
        of taking that action.
        """
        o_trans = self.feature_transformer.transform(o)
        return self.Q[o_trans]

    def update(self, otm1, atm1, r, ot, at):
        """
        Performs TD(0) update on the Q value
        """
        otm1_trans = self.feature_transformer.transform(otm1)
        ot_trans = self.feature_transformer.transform(ot)
        G = r + self.gamma*self.Q[ot_trans, at]
        self.Q[otm1_trans, atm1] += self.alpha*(G - self.Q[otm1_trans, atm1])

    def sample_action(self, o, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return self.best_action(o)

    def best_action(self, o):
        return np.argmax(self.predict(o))
