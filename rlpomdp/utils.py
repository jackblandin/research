import numpy as np
import matplotlib.pyplot as plt


def play_one(env, model, eps, verbose=False):
    """
    Plays one episode in the environment using the model.

    Parameters
    ----------
    env : gym.Env
        OpenAI Gym environment.
    model : object
        RL model (e.g. QLearner).
    eps : float
        Epsilon for epsilon-greedy strategy.
    verbose : boolean
        Whether or not to print out additional debugging info.

    Returns
    -------
    float
        Total reward accumulated during episode.
    """
    ot = env.reset()
    done = False
    total_episode_reward = 0
    iters = 0
    if verbose:
        print('{:<40} | {:<11} | {:<10} | {:<5} | {:<11} | {:<10}'.format(
            'o_t-n,...,o_t-1', 'o_t-1', 'a_t-1', 'r', 'o_t', 'a_t'))
        print('-'*100)
    while not done:
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
            if hasattr(model, 'last_n_obs'):
                _last_n_obs = str(
                    [env.translate_obs(o) for o in model.last_n_obs])
            else:
                _last_n_obs = ''
            print('{:<40} | {:<11} | {:<10} | {:<5} | {:<11} | {:<10}'.format(
                _last_n_obs, _otm1, _atm1, r, _ot, _at))
        iters += 1
    return total_episode_reward


def plot_running_avg(totalrewards, figsize=(20, 5)):
    """
    Plots the total reward running average.

    Parameters
    ----------
    totalrewards : list or array-like
        Array where index is episode number and value is total rewards
        accumulated during the episode.

    figsize : tuple (length two)
        Width, height of the figure displaying the running average.

    Returns
    -------
    None
    """
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(running_avg)
    ax.set_title("Running Average")
