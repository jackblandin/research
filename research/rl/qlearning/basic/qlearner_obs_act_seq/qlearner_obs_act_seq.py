import numpy as np
import pandas as pd
from IPython.display import display


class QLearnerObsActSeq:

    def __init__(self, env, feature_transformer, initial_alpha=.1, gamma=.9,
                 alpha_decay=0, seq_len=3):
        """
        Started 06/27/2019
        Q Learner with all combinations of seq_len action-observations as state
        space.

        Parameters
        ----------
        env : gym.Env
            OpenAI Gym environment.
        feature_transformer : object
            Object that transforms raw state/observations into the Q function's
            state representation. Must have a `transform()` instance method.
        initial_alpha : float
            Learning rate.
        gamma : float
            Discount factor.
        alpha_decay : float, default 0
            Learning rate alpha will decay at 1/_n_updates**_alpha_decay.
        seq_len : int
            Number of sequential action-observations to use as the state.

        Attributes
        ----------
        feature_transformer : object
            Transforms raw state into representation, usually a reduced one.
        Q : 2D numpy array
            Q <state,value> matrix where
                - axis0 index is transformed action-observation
                - axis1 index is action
                - value is Q value.
            E.g. Q[1][2] represents the Q value for taking action 2 for
            action-observation index 1.
        _n_updates : int
            Number of updates made to Q matrix.
        history : np.ndarray, shape (seq_len, 3)
            Each row is [timestep, action, observation].
        Q_update_counts_ : np.ndarray
            Keeps track of the number of times each Q value is updated.
        """
        self.env = env
        self.feature_transformer = feature_transformer
        self.initial_alpha = initial_alpha
        self.gamma = gamma
        self._alpha_decay = alpha_decay
        self.seq_len = seq_len
        num_obs = env.observation_space.n
        num_actions = env.action_space.n
        exp = seq_len
        num_states = 1
        while exp > 0:
            num_states += (num_obs*num_actions)**exp
            exp = exp - 1
        self.Q = np.zeros((num_states, num_actions))
        self._n_updates = 0
        self.history = np.empty((self.seq_len, 3), dtype=object)
        self.Q_update_counts_ = np.zeros((num_states, num_actions))

    def predict(self, t, ot):
        """
        Uses the input observation (ot) and the last <seq_len-1> observations
        + actions as the input to the Q matrix.

        Parameters
        ----------
        t : int
            Timestep.
        ot : list or array-like <int>
            Observation at time t.

        Returns
        -------
        An array where index is an action, and values are the Q values
        of taking that action.
        """
        # Build the staggered [[ot1, at0], [ot2, at1], ...] action-observation
        # sequences, using the input `ot` as the latest observation. Note that
        # `at` will be None.
        seq, _ = self._stagger_action_obs_seq(t, self.history, ot)

        # Transform action-observation sequence
        seq_t = self.feature_transformer.transform(seq)

        # Lookup Q values of transformed action-observation sequence
        return self.Q[seq_t]

    def update(self, t, ot, at, r, ot1, at1):
        """
        Performs TD(0) update on the model using sequences of
        observation-actions as states. Note that this does NOT use experience
        replay, which is how we are able to avoid storing more than the last
        <seq_len> nsequences of action-observations.

        Parameters
        ----------
        t : int
            Timestep.
        ot : list or array-like
            (UNUSED) Observation observed after taking action at-1.
        at : int
            Action taken after observing ot.
        r : float
            Observed reward after taking action at.
        o1 : list or array-like
            Observation observed after taking action at.
        a1 : Action taken after observing ot.

        Returns
        -------
        None
        """
        # Build the staggered [[ot1, at0], [ot2, at1], ...] action-observation
        # sequences, using the input `ot` as the latest observation. Note that
        # `at` will be None. This will be used to compute the current Q values.
        seqt, ht = self._stagger_action_obs_seq(t, self.history, ot)

        # Build staggered sequence at next time stamp. This will be used to
        # Compute the "G" Q-values.
        seqt1, ht1 = self._stagger_action_obs_seq(t, ht, ot1)
        ht1[-1, 1] = at

        if t > self.seq_len + 1:

            if None in seqt:
                display(seqt)
                raise ValueError('seqt contains Nones at time {}'.format(t))

            # Transform seqt, seqt1
            seqt_t = self.feature_transformer.transform(seqt)
            seqt1_t = self.feature_transformer.transform(seqt1)

            # Compute alpha (if there's nonzero alpha decay)
            alpha = self.initial_alpha / (self._n_updates+1)**self._alpha_decay

            # TD(0) update using observation-action sequences as states.
            Qt = self.Q[seqt_t, at]
            Qt1 = self.Q[seqt1_t, at1]
            G = r + self.gamma * Qt1
            self.Q[seqt_t, at] += alpha * (G - Qt)
            self._n_updates += 1
            self.Q_update_counts_[seqt_t, at] += 1

        # Store observation-action pair in history attribute.
        self._add_history(t, ot, at)

    def sample_action(self, t, o, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return self.best_action(t, o)

    def best_action(self, t, o):
        """
        Returns the action with the highest q value, given an observation.

        Parameters
        ----------
        o : list or array-like <int>
            A single observation.

        Returns
        -------
        int
            Best action given observation.
        """
        # Can't predict correctly if we don't have enough action-observation
        # sequences stored.
        if self._n_updates < self.seq_len:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(t, o))

    def _stagger_action_obs_seq(self, t, historyt, ot):
        """
        Builds the staggered [[ot1, at0], [ot2, at1], ...] action-observation
        sequences, using the input `ot` as the latest observation. Note that
        `at` will be None.

        Parameters
        ----------
        t : int
            Timestep.
        historyt : np.ndarray, shape (seq_len, 3)
            History at time t.
        ot : list or array-like <int>
            Single observation.

        Returns
        -------
        tuple, len 2
            np.ndarray, shape (seq_len+1, 2)
                Raw observation-action sequence.
            np.ndarray, shape (seq_len, 3)
                History used to compute observaiton-action sequence.
        """
        ht = historyt.copy()

        # Add observation as new row in temp copy of history
        ht1 = np.empty((self.seq_len+1, 3), dtype=object)
        ht1[0:self.seq_len] = ht[0:self.seq_len]

        # Action has not yet been chosen, so it's None
        ht1[self.seq_len] = np.array([t+1, None, ot])

        # Need observations and actions staggered w. regard to timestep
        seq = np.empty((len(ht1)-1, 2), dtype=object)
        for i, (o, a) in enumerate(zip(ht1[1:, 2], ht1[0:-1, 1])):
            seq[i] = np.array([o, a], dtype=object).reshape(-1,)

        assert seq.shape == (self.seq_len, 2)
        assert ht1.shape == (self.seq_len+1, 3)

        return seq, ht1

    def _add_history(self, t, ot, at):
        """Appends an timestep, action, observation to history.

        If len(history) > seq_len, then the first item in the list is popped
        (fifo).

        Parameters
        ----------
        t : int
            Timestep.
        ot : list<int>
            Observation observed after taking action at.
        at : int
            Action taken at time t.

        Returns
        -------
        None
        """
        if len(self.history) == self.seq_len:
            self.history = np.delete(self.history, 0, axis=0)
        self.history = np.vstack([self.history, np.array([t, at, ot],
                                                         dtype=object)])

    def to_df(self):
        """
        DataFrame representation of model Q values.

        Returns
        -------
        pandas.DataFrame
            A dataframe showing the Q values of each state/action.
        """
        env_class = self.env.__class__.__name__

        if env_class == 'TigerEnv':
            actions = [0, 1, 2]
            actions_ = [self.env.translate_action(a1) for a1 in actions]
            obs_act_seqs = self.feature_transformer.inverse_lookup_.values()
            rows = []
            for obs_act_seq in obs_act_seqs:
                if len(obs_act_seq) == 0:
                    continue
                row = []
                for i, obs_act in enumerate(obs_act_seq):
                    o0, a0 = obs_act
                    o0_ = self.env.translate_obs(o0)
                    a0_ = self.env.translate_action(a0)
                    row.append(a0_)
                    row.append(o0_)
                best_Q = None
                best_action = None
                for a1 in actions:
                    oa_seq_t = self.feature_transformer.transform(obs_act_seq)
                    Q = self.Q[oa_seq_t, a1].round(2)
                    Q_update_count = int(self.Q_update_counts_[oa_seq_t, a1])
                    Qval = str(Q)
                    if best_Q is None or Q > best_Q:
                        best_Q = Q
                        best_action = a1
                    row.append(Qval)
                    row.append(Q_update_count)
                if best_Q == 0:
                    continue
                row.append(actions_[best_action])
                rows.append(row)

            update_counts_ = [a + ' UPDATE COUNT' for a in actions_]
            Q_columns = []
            for a, u in zip(actions_, update_counts_):
                Q_columns.append(a + ' Q VALUE')
                Q_columns.append(u)
            columns = []
            for i in range(self.seq_len):
                columns.append('a_t-{}'.format(self.seq_len-i))
                columns.append('o_t-{}'.format(self.seq_len-i-1))
            columns += Q_columns + ['BEST ACTION']
            df = pd.DataFrame(rows, columns=columns)
            return df
        else:
            raise ValueError('Don\'t know how to represent this env.')


def play_one(env, model, eps, verbose=False):
    """Plays one episode in the environment using the model.

    Responsible for updating the selecting and taking action, updating the
    model, and displaying learning progress via accumulated rewards.

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
    t = 0

    if verbose:
        print('{:<5} | {:<11} | {:<10} | {:<5} | {:<11} | {:<10}'.format(
            't', 'o_t-1', 'a_t-1', 'r', 'o_t', 'a_t'))
        print('-'*100)

    while not done:
        otm1 = ot

        # Select action based on otm1 (and history which is stored in model)
        atm1 = model.sample_action(t, otm1, eps)

        # Take action
        ot, r, done, info = env.step(atm1)

        # Select best action for next timestep
        at = model.best_action(t, ot)

        # Update Q values
        model.update(t, otm1, atm1, r, ot, at)

        total_episode_reward += r

        if verbose:
            _otm1 = env.translate_obs(otm1)
            _atm1 = env.translate_action(atm1)
            _ot = env.translate_obs(ot)
            _at = env.translate_action(at)
            print('{:<5} | {:<11} | {:<10} | {:<5} | {:<11} | {:<10}'.format(
                t, _otm1, _atm1, r, _ot, _at))
        t += 1
    return total_episode_reward
