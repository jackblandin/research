def env_translate_obs(obs):
    """
    This should only be used for the Tiger ENV.

    Parameters
    ----------
    obs : list or array-like
        The observation to be translated.

    Returns
    -------
    str
        A representation of the observation in English.
    """
    if obs[0] == 1:
        return 'GROWL_LEFT'
    elif obs[1] == 1:
        return 'GROWL_RIGHT'
    elif obs[2] == 1:
        return 'START'
    elif obs[3] == 1:
        return 'END'
    else:
        raise ValueError('Invalid observation: '.format(obs))


def env_translate_action(action):
    """
    This should only be used for the Tiger ENV.

    Parameters
    ----------
    action : int
        The action to be translated.

    Returns
    -------
    str
        A representation of the action in English.
    """
    ACTION_OPEN_LEFT = 0
    ACTION_OPEN_RIGHT = 1
    ACTION_LISTEN = 2
    ACTION_MAP = {
        ACTION_OPEN_LEFT: 'OPEN_LEFT',
        ACTION_OPEN_RIGHT: 'OPEN_RIGHT',
        ACTION_LISTEN: 'LISTEN',
    }
    return ACTION_MAP[action]
