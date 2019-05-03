import numpy as np
import matplotlib.pyplot as plt


def play_one(env, model, eps):
    ot = env.reset()
    done = False
    total_episode_reward = 0
    iters = 0
    while not done and iters < 10:
        otm1 = ot
        atm1 = model.sample_action(otm1, eps)
        ot, r, done, info = env.step(atm1)
        at = model.best_action(ot)
        model.update(otm1, atm1, r, ot, at)
        total_episode_reward += r
    return total_episode_reward


def plot_running_avg(totalrewards, figsize=(20, 5)):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(running_avg)
    ax.set_title("Running Average")
