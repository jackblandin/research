# System
import sys
import os
from datetime import datetime
import numpy as np

# 3rd party
import gym
from gym import wrappers
import matplotlib.pyplot as plt

# Local
from utils import plot_running_avg

class FeatureTransformer:
    def __init__(self, env):
        # TODO
        self.featurizer = None

    def transform(self, observations):
        return self.featurizer.transform(observations)


class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.model = None
        self.feature_transformer = feature_transformer

    def predict(self, s):
        X = self.feature_transformer.transform(s)
        return self.model.predict(X)

    def update(self, s, a, G):
        # X = self.feature_transformer.transform(s)
        # TODO
        pass

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))


def play_one(env, model, eps, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done and iters < 2000:
        # TODO
        pass
    return totalreward


def main():
    env = gym.make('CartPole-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft)
    gamma = 0.99

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 500
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 1.0/np.sqrt(n+1)
        totalreward = play_one(env, model, eps, gamma)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print("episode:", n, "total reward:", totalreward, "eps:", eps,
                  "avg reward (last 100):",
                  totalrewards[max(0, n-100):(n+1)].mean())

    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)


if __name__ == '__main__':
    main()
