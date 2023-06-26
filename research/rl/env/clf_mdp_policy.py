import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class ClassificationMDPPolicy(BaseEstimator, ClassifierMixin):
    """
    A Scikit-Learn compatible wrapper for a Classification MDP policy.

    Parameters
    ----------
    mdp : ClassificationMDP
        The classification mdp instance.
    pi : numpy.array<int>
        The optimal policy (optimal action once `y` is known).
    clf : sklearn.BaseEstimator
        Binary classifier used for predicting `y` from `X`. It needs to be
        fitted already, and it needs to contain the full preprocessing of
        inputs.
    default_action : int
        The default action to use if the state lookup fails.

    Attributes
    ----------
    reward_weights : list<float>
        Weights for each objective component.
    pi_df_ : pandas.DataFrame
        Policy represented as a dataframe.
    """

    def __init__(self, mdp, pi, clf, default_action=0):
        self.mdp = mdp
        self.pi = pi
        self.clf = clf
        self.default_action = default_action
        self._construct_pi_df()

    def fit(self, X, y):
        """
        Pass through. Doesn't do anything.

        Parameters
        ----------
        X : pandas.DataFrame
            Classification input.
        y : pandas.Series<int>
            Binary target variable.

        Returns
        -------
        None
        """
        pass

    def predict(self, X, y=None):
        """
        If `y` is None, then predicts `y` from `X`, then returns the optimal
        action for that value of `(X, y)`. Sort of like a two-step POMDP.

        If `y` is provided, then does not predict `y`.

        I.e. here is some crude pseudocode reprsenting what's actually
        happening:
            ```
            if y is None:
                y = predict(X)

            a = pi(X, y)
            return a
            ```
        Parameters
        ---------
        X : pandas.DataFrame
            Input data.
        y : pandas.Series
            label data.

        Returns
        -------
        actions : numpy.array<int>, len(len(X))
            The "predictions", actually the actions from the Clf MDP.
        """
        X = X.copy()
        df = pd.DataFrame(X)
        # By using `predict_proba` and inserting randomness, we ensure that the
        # assumed y values are not always the majority.
        # df['y'] = (
        #     self.clf.predict_proba(X)[:,0] >= np.random.rand(len(X))
        # ).astype(int)
        if y is None:
            df['y'] = self.clf.predict(df)
        else:
            df['y'] = y

        actions = np.zeros(len(X))

        # Get rid of any unused columns otherwise the state lookup breaks.
        df = df[self.mdp.x_cols + ['z', 'y']]

        # Reduce state input using fitted state_reducer_
        for x in self.mdp.state_reducer_.keys():
            for x_val in self.mdp.state_reducer_[x]:
                df.loc[df[x] == x_val, x] = self.mdp.state_reducer_[x][x_val]

        n_state_lookup_errors = 0

        for i, (idx, row) in enumerate(df.iterrows()):
            try:
                state = self.mdp.reduced_state_lookup_[tuple(row)]
                actions[i] = self.pi[state]
            except KeyError as e:
                logging.debug('\tState Lookup Error: ' + str(e))
                logging.debug(f"\tUsing default action: {self.default_action}")
                actions[i] = self.default_action
                n_state_lookup_errors += 1
        log_msg = f"""
            \t\tThere were {n_state_lookup_errors} state lookup errors when trying
            \t\tto set the optimal action for the input dataset. There are
            \t\t{len(df)} total input rows. So {n_state_lookup_errors}/{len(df)}
            \t\tforced to use the default action ({self.default_action}).
            """
        logging.debug(log_msg)
        return actions

    def _construct_pi_df(self):
        """
        Adds the policy actions as a column to the self.mdp.ldf_ dataframe.

        Sets pi_df_
        ----------
        pi_df_ : pandas.DataFrame
            Policy represented as a dataframe.
        """
        pi_df = self.mdp.ldf_.iloc[:, :-2].drop_duplicates().copy()
        pi_df['a'] = self.predict(pi_df)
        self.pi_df_ = pi_df
        return None
