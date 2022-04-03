import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # noqa


def play_episode(env, model, eps):
    """
    Plays a single episode. During play, the model is updated, and the total
    reward is accumulated and returned.
    Parameters
    ----------
    env : gym.Env
        Environment.
    model : <TBD>
        Model instance.
    eps : numeric
        Epsilon used in epsilon-greedy.
    Returns
    -------
    numeric
        Total reward accumualted during episode.
    """
    obs = env.reset()
    done = False
    totalreward = 0
    timestep = 0

    while not done:

        # Choose an action based on current observation.
        action = model.select_action(obs, eps)
        prev_obs = obs

        # Take chosen action.
        obs, reward, done, _ = env.step(action)

        totalreward += reward

        # Update the model
        model.add_experience(prev_obs, action, reward, obs, done)
        model.train(timestep)

        timestep += 1

    return totalreward


def play_n_episodes(env, model, n, use_eps=True):
    totalrewards = np.zeros(n)

    if n > 10:
        window = int(n/10)
    else:
        window = 1

    for _n in range(n):

        if not use_eps:
            eps = 0
        else:
            if _n >= (n - window):
                eps = 0
            else:
                eps = 1.0/(_n+1)**.2

        totalreward = play_episode(env, model, eps)
        totalrewards[_n] = totalreward
        if _n % window == 0:
            ravg = _running_avg(totalrewards, _n, window)
            print('episode: {:,}, total reward: {:,.2f}, eps: {:.3f}, avg '
                  'reward last {:,}: {:.3f}'.format(_n, totalreward, eps,
                                                    window, ravg))

    print('\nTotal steps: {:,}'.format(len(totalrewards)))
    print('Avg cumulative reward: {:,.3f}'.format(totalrewards.mean()))
    print('Avg reward for last {:,} episodes: {:,.3f}'.format(
        window, totalrewards[int(-1*(n/10)):].mean()))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 5))
    ax0.plot(totalrewards)
    ax0.set_title("Rewards")
    _plot_running_avg(ax1, totalrewards, window)
    return ax0, ax1


def _running_avg(totalrewards, t, window):
    return totalrewards[max(0, t-window):(t+1)].mean()


def _plot_running_avg(ax, totalrewards, window):
    N = len(totalrewards)
    ravg = np.empty(N)
    for t in range(N):
        ravg[t] = _running_avg(totalrewards, t, window)
    ax.plot(ravg)
    ax.set_title('Running Average')
