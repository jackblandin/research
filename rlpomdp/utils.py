import numpy as np
import matplotlib.pyplot as plt


def play_one(env, model, eps, max_iters=10, verbose=False):
    ot = env.reset()
    done = False
    total_episode_reward = 0
    iters = 0
    if verbose:
        print('{:<11} | {:<10} | {:<5} | {:<11} | {:<10}'.format(
                'o_t-1', 'a_t-1', 'r', 'o_t', 'a_t'))
        print('-'*55)
    while not done and iters < max_iters:
        otm1 = ot
        atm1 = model.sample_action(otm1, eps)
        ot, r, done, info = env.step(atm1)
        at = model.best_action(ot)
        model.update(otm1, atm1, r, ot, at)
        total_episode_reward += r
        if verbose:
            _otm1 = env.translate_obs(otm1)
            _atm1 = env.translate_action(atm1)
            _ot = env.translate_obs(ot)
            _at = env.translate_action(at)
            print('{:<11} | {:<10} | {:<5} | {:<11} | {:<10}'.format(
                _otm1, _atm1, r, _ot, _at))
    return total_episode_reward


def plot_running_avg(totalrewards, figsize=(20, 5)):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(running_avg)
    ax.set_title("Running Average")
