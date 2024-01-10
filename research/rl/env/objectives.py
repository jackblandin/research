import itertools
import logging
import numpy as np
import pandas as pd


class OptimizationProblem():

    def __init__(self, name, c, A_eq=None, b_eq=None, b_ub=None, A_ub=None):
        self.name = name
        self.c = c
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.b_ub = b_ub
        self.A_ub = A_ub

class ObjectiveSplit():

    def __init__(self, name, parent, c, b_ub=None, A_ub=None):
        self.name = name
        self.parent = parent
        self.c = c
        self.b_ub = b_ub
        self.A_ub = A_ub

class Objective():

    def __init__(self):
        pass

    def fit(self, ldf):
        raise NotImplementedError()

    def to_splits(self):
        raise NotImplementedError()

    def compute_feat_exp(self, demo):
        raise NotImplementedError()


class LinearObjective(Objective):

    def __init__(self):
        self.n_splits = 1
        super().__init__()

    def fit(self, ldf):
        self.b_ub_row_ = None
        self.c_ = self._construct_reward(ldf)
        return self

    def to_splits(self):
        """
        Returns objective as one or more splits.

        Parameters
        ----------
        None

        Returns
        -------
        array<ObjectiveSplit>
        """
        split = ObjectiveSplit(
            name=self.name,
            parent=self,
            c=self.c_,
            A_ub=None,
            b_ub=None,
        )
        return [split]

    def _construct_reward(self):
        raise NotImplementedError


class AbsoluteValueObjective(Objective):

    def __init__(self):
        self.n_splits = 2
        super().__init__()

    def fit(self, ldf):
        self.b_ub_row_ = 0  # Numb of constraints
        self.A_ub_row__split1_ = self._compute_A_ub_row__split1(ldf)
        self.A_ub_row__split2_ = self._compute_A_ub_row__split2(ldf)
        self.c__split1_ = self._construct_reward__split1(ldf)
        self.c__split2_ = self._construct_reward__split2(ldf)
        return self

    def to_splits(self):
        """
        Returns objective as one or more splits.

        Parameters
        ----------
        None

        Returns
        -------
        array<ObjectiveSplit>
        """
        split1 = ObjectiveSplit(
            name=f"{self.name} Split1",
            parent=self,
            c=self.c__split1_,
            b_ub=self.b_ub_row_,
            A_ub=self.A_ub_row__split1_,
        )
        split2 = ObjectiveSplit(
            name=f"{self.name} Split2",
            parent=self,
            c=self.c__split2_,
            b_ub=self.b_ub_row_,
            A_ub=self.A_ub_row__split2_,
        )
        return [split1, split2]

    def _compute_A_ub_row__split1(self, ldf):
        raise NotImplementedError()

    def _compute_A_ub_row__split2(self, ldf):
        raise NotImplementedError()

    def _construct_reward__split1(self, ldf):
        raise NotImplementedError()

    def _construct_reward__split2(self, ldf):
        raise NotImplementedError()


class AccuracyObjective(LinearObjective):

    def __init__(self):
        self.name = 'Acc'
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        mu = np.mean(demo['yhat'] == demo['y'])

        return mu

    def _construct_reward(self, ldf):
        """
        Constructs the reward function when the objective is accuracy.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        ------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        p_all = ldf['mu0'].sum() / 2
        ldf['r'] = (ldf['yhat'] == ldf['y']).astype(float) / p_all
        c = -1 * ldf['r']  # Negative since maximizing not minimizing

        # Check c sums to 1
        assert (c.sum() + 1) < 1e-8

        return c

class AccuracyParityObjective(AbsoluteValueObjective):

    def __init__(self):
        self.name = 'AccPar'
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_y_giv_z_eq_0 = (
            ((demo['yhat'] == demo['y']) & (demo['z'] == 0)).sum()
            / (demo['z'] == 0).sum()
        )
        p_yhat_eq_y_giv_z_eq_1 = (
            ((demo['yhat'] == demo['y']) & (demo['z'] == 1)).sum()
            / (demo['z'] == 1).sum()
        )
        mu = 1 - max([
            p_yhat_eq_y_giv_z_eq_0 - p_yhat_eq_y_giv_z_eq_1,
            p_yhat_eq_y_giv_z_eq_1 - p_yhat_eq_y_giv_z_eq_0,
        ])
        if np.isnan(mu):
            mu = 1

        return mu

    def _compute_A_ub_row__split1(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=y|z=0) >= P(yhat=y|z=1)
            ```
        which is Accuracy Parity opt_problem 1.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        n_actions = 2
        ldf = ldf.copy()
        filt__yhaty_z0 = (ldf['z'] == 0) & (ldf['yhat'] == ldf['y'])
        filt__yhaty_z1 = (ldf['z'] == 1) & (ldf['yhat'] == ldf['y'])
        p_z0 = ldf[ldf['z'] == 0]['mu0'].sum() / 2
        p_z1 = ldf[ldf['z'] == 1]['mu0'].sum() / 2
        ldf['A_ub'] = 0.0
        ldf.loc[filt__yhaty_z0, 'A_ub'] = -1 / p_z0
        ldf.loc[filt__yhaty_z1, 'A_ub'] =  1 / p_z1

        return ldf['A_ub']

    def _compute_A_ub_row__split2(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=y|z=0) <= P(yhat=y|z=1)
            ```
        which is Accuracy Parity opt_problem 1.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        n_actions = 2
        ldf = ldf.copy()
        filt__yhaty_z0 = (ldf['z'] == 0) & (ldf['yhat'] == ldf['y'])
        filt__yhaty_z1 = (ldf['z'] == 1) & (ldf['yhat'] == ldf['y'])
        p_z0 = ldf[ldf['z'] == 0]['mu0'].sum() / 2
        p_z1 = ldf[ldf['z'] == 1]['mu0'].sum() / 2
        ldf['A_ub'] = 0.0
        ldf.loc[filt__yhaty_z0, 'A_ub'] =  1 / p_z0
        ldf.loc[filt__yhaty_z1, 'A_ub'] = -1 / p_z1

        return ldf['A_ub']

    def _construct_reward__split1(self, ldf):
        """
        Constructs the reward function for Accuracy Parity  opt_problem 1.
        opt_problem 1 is when we constrain P(yhat=y|z=0) >= P(yhat=y|z=1), in
        which case the reward penalizes giving the Z=0 group the correct
        prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhaty_z0 = (ldf['z'] == 0) & (ldf['yhat'] == ldf['y'])
        filt__yhaty_z1 = (ldf['z'] == 1) & (ldf['yhat'] == ldf['y'])
        p_z0 = ldf[ldf['z'] == 0]['mu0'].sum() / 2
        p_z1 = ldf[ldf['z'] == 1]['mu0'].sum() / 2
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhaty_z0, 'r'] = -1 / p_z0
        ldf.loc[filt__yhaty_z1, 'r'] =  1 / p_z1
        c = -1 * ldf['r']  # Negative since maximizing not minimizing

        return c

    def _construct_reward__split2(self, ldf):
        """
        Constructs the reward function for Accuracy Parity  opt_problem 1.
        opt_problem 1 is when we constrain P(yhat=y|z=0) <= P(yhat=y|z=1), in
        which case the reward penalizes giving the Z=1 group the correct
        prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhaty_z0 = (ldf['z'] == 0) & (ldf['yhat'] == ldf['y'])
        filt__yhaty_z1 = (ldf['z'] == 1) & (ldf['yhat'] == ldf['y'])
        p_z0 = ldf[ldf['z'] == 0]['mu0'].sum() / 2
        p_z1 = ldf[ldf['z'] == 1]['mu0'].sum() / 2
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhaty_z0, 'r'] =  1 / p_z0
        ldf.loc[filt__yhaty_z1, 'r'] = -1 / p_z1
        c = -1 * ldf['r']  # Negative since maximizing not minimizing

        return c


class DemographicParityObjective(AbsoluteValueObjective):

    def __init__(self):
        self.name = 'DemPar'
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_1_giv_z_eq_0 = (
            ((demo['yhat'] == 1) & (demo['z'] == 0)).sum()
            / (demo['z'] == 0).sum()
        )
        p_yhat_eq_1_giv_z_eq_1 = (
            ((demo['yhat'] == 1) & (demo['z'] == 1)).sum()
            / (demo['z'] == 1).sum()
        )
        mu = 1 - max([
            p_yhat_eq_1_giv_z_eq_0 - p_yhat_eq_1_giv_z_eq_1,
            p_yhat_eq_1_giv_z_eq_1 - p_yhat_eq_1_giv_z_eq_0,
        ])

        if np.isnan(mu):
            mu = 1

        return mu

    def _compute_A_ub_row__split1(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=1|z=0) >= P(yhat=1|z=1)
            ```
        which is Demographic Parity opt_problem 1.

        Parameters
        ----------
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        n_actions = 2
        ldf = ldf.copy()
        filt__yhat1_z0 = (ldf['z'] == 0) & (ldf['yhat'] == 1)
        filt__yhat1_z1 = (ldf['z'] == 1) & (ldf['yhat'] == 1)
        p_z0 = ldf[ldf['z'] == 0]['mu0'].sum() / 2
        p_z1 = ldf[ldf['z'] == 1]['mu0'].sum() / 2
        ldf['A_ub'] = 0.0
        ldf.loc[filt__yhat1_z0, 'A_ub'] = -1 / p_z0
        ldf.loc[filt__yhat1_z1, 'A_ub'] =  1 / p_z1

        return ldf['A_ub']

    def _compute_A_ub_row__split2(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=1|z=1) >= P(yhat=1|z=0)
            ```
        which is Demographic Parity opt_problem 2.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.


        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        n_actions = 2
        ldf = ldf.copy()
        filt__yhat1_z0 = (ldf['z'] == 0) & (ldf['yhat'] == 1)
        filt__yhat1_z1 = (ldf['z'] == 1) & (ldf['yhat'] == 1)
        p_z0 = ldf[ldf['z'] == 0]['mu0'].sum() / 2
        p_z1 = ldf[ldf['z'] == 1]['mu0'].sum() / 2
        ldf['A_ub'] = 0.0
        ldf.loc[filt__yhat1_z0, 'A_ub'] =  1 / p_z0
        ldf.loc[filt__yhat1_z1, 'A_ub'] = -1 / p_z1

        return ldf['A_ub']

    def _construct_reward__split1(self, ldf):
        """
        Constructs the reward function for Demographic Parity opt_problem 1.
        opt_problem 1 is when we constrain P(yhat=1|z=0) >= P(yhat=1|z=1), in
        which case the reward penalizes giving the Z=0 group the positive
        prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_z0 = (ldf['z'] == 0) & (ldf['yhat'] == 1)
        filt__yhat1_z1 = (ldf['z'] == 1) & (ldf['yhat'] == 1)
        p_z0 = ldf[ldf['z'] == 0]['mu0'].sum() / 2
        p_z1 = ldf[ldf['z'] == 1]['mu0'].sum() / 2
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_z0, 'r'] = -1 / p_z0
        ldf.loc[filt__yhat1_z1, 'r'] = 1 / p_z1
        c = -1 * ldf['r']  # Negative since maximizing not minimizing

        return c

    def _construct_reward__split2(self, ldf):
        """
        Constructs the reward function for Demographic Parity opt_problem 2.
        opt_problem 2 is when we constrain P(yhat=1|z=1) >= P(yhat=1|z=0), in
        which case the reward penalizes giving the Z=1 group the positive
        prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_z0 = (ldf['z'] == 0) & (ldf['yhat'] == 1)
        filt__yhat1_z1 = (ldf['z'] == 1) & (ldf['yhat'] == 1)
        p_z0 = ldf[ldf['z'] == 0]['mu0'].sum() / 2
        p_z1 = ldf[ldf['z'] == 1]['mu0'].sum() / 2
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_z0, 'r'] =  1 / p_z0
        ldf.loc[filt__yhat1_z1, 'r'] = -1 / p_z1
        c = -1 * ldf['r']  # Negative since maximizing not minimizing

        return c


class EqualOpportunityObjective(AbsoluteValueObjective):

    def __init__(self):
        self.name = 'EqOpp'
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_1_giv_y_eq_1_z_eq_0 = (
            ((demo['yhat'] == 1) & (demo['y'] == 1) & (demo['z'] == 0)).sum()
            / ((demo['y'] == 1) & (demo['z'] == 0)).sum()
        )
        p_yhat_eq_1_giv_y_eq_1_z_eq_1 = (
            ((demo['yhat'] == 1) & (demo['y'] == 1) & (demo['z'] == 1)).sum()
            / ((demo['y'] == 1) & (demo['z'] == 1)).sum()
        )
        mu = 1 - max([
            p_yhat_eq_1_giv_y_eq_1_z_eq_0 - p_yhat_eq_1_giv_y_eq_1_z_eq_1,
            p_yhat_eq_1_giv_y_eq_1_z_eq_1 - p_yhat_eq_1_giv_y_eq_1_z_eq_0,
        ])
        if np.isnan(mu):
            mu = 1

        return mu

    def _compute_A_ub_row__split1(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=1|y=1,z=0) >= P(yhat=1|y=1,z=1)
            ```
        which is Equal Opportunity opt_problem 1.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        n_actions = 2
        ldf = ldf.copy()
        filt__yhat1_y1_z0 = (ldf['z'] == 0) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        filt__yhat1_y1_z1 = (ldf['z'] == 1) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        p_z0_y1 = ldf[(ldf['z'] == 0) & (ldf['y'] == 1)]['mu0'].sum() / 2
        p_z1_y1 = ldf[(ldf['z'] == 1) & (ldf['y'] == 1)]['mu0'].sum() / 2
        ldf['A_ub'] = 0.0
        ldf.loc[filt__yhat1_y1_z0, 'A_ub'] = -1 / p_z0_y1
        ldf.loc[filt__yhat1_y1_z1, 'A_ub'] =  1 / p_z1_y1

        return ldf['A_ub']

    def _compute_A_ub_row__split2(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=1|y=1,z=1) >= P(yhat=1|y=1,z=0)
            ```
        which is Equal Opportunity opt_problem 2.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        n_actions = 2
        ldf = ldf.copy()
        filt__yhat1_y1_z0 = (ldf['z'] == 0) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        filt__yhat1_y1_z1 = (ldf['z'] == 1) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        p_z0_y1 = ldf[(ldf['z'] == 0) & (ldf['y'] == 1)]['mu0'].sum() / 2
        p_z1_y1 = ldf[(ldf['z'] == 1) & (ldf['y'] == 1)]['mu0'].sum() / 2
        ldf['A_ub'] = 0.0
        ldf.loc[filt__yhat1_y1_z0, 'A_ub'] =  1 / p_z0_y1
        ldf.loc[filt__yhat1_y1_z1, 'A_ub'] = -1 / p_z1_y1

        return ldf['A_ub']

    def _construct_reward__split1(self, ldf):
        """
        Constructs the reward function for Equal Opportunity  opt_problem 1.
        opt_problem 1 is when we constrain P(yhat=1|y=1,z=0) >= P(yhat=1|y=1,z=1), in
        which case the reward penalizes giving the Z=0 group the positive
        prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_y1_z0 = (ldf['z'] == 0) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        filt__yhat1_y1_z1 = (ldf['z'] == 1) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        p_y1_z0 = ldf[(ldf['y'] == 1) & (ldf['z'] == 0)]['mu0'].sum() / 2
        p_y1_z1 = ldf[(ldf['y'] == 1) & (ldf['z'] == 1)]['mu0'].sum() / 2
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_y1_z0, 'r'] = -1 / p_y1_z0
        ldf.loc[filt__yhat1_y1_z1, 'r'] =  1 / p_y1_z1
        c = -1 * ldf['r']  # Negative since maximizing not minimizing

        return c

    def _construct_reward__split2(self, ldf):
        """
        Constructs the reward function for Demographic Parity opt_problem 1.
        opt_problem 1 is when we constrain P(yhat=1|y=1,z=0) >= P(yhat=1|y=1,z=1), in
        which case the reward penalizes giving the Z=0 group the positive
        prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_y1_z0 = (ldf['z'] == 0) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        filt__yhat1_y1_z1 = (ldf['z'] == 1) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        p_y1_z0 = ldf[(ldf['y'] == 1) & (ldf['z'] == 0)]['mu0'].sum() / 2
        p_y1_z1 = ldf[(ldf['y'] == 1) & (ldf['z'] == 1)]['mu0'].sum() / 2
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_y1_z0, 'r'] =  1 / p_y1_z0
        ldf.loc[filt__yhat1_y1_z1, 'r'] = -1 / p_y1_z1
        c = -1 * ldf['r']  # Negative since maximizing not minimizing

        return c


class FalsePositiveRateParityObjective(AbsoluteValueObjective):

    def __init__(self):
        self.name = 'FPRPar'
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_1_giv_y_eq_0_z_eq_0 = (
            ((demo['yhat'] == 1) & (demo['y'] == 0) & (demo['z'] == 0)).sum()
            / ((demo['y'] == 0) & (demo['z'] == 0)).sum()
        )
        p_yhat_eq_1_giv_y_eq_0_z_eq_1 = (
            ((demo['yhat'] == 1) & (demo['y'] == 0) & (demo['z'] == 1)).sum()
            / ((demo['y'] == 0) & (demo['z'] == 1)).sum()
        )
        mu = 1 - max([
            p_yhat_eq_1_giv_y_eq_0_z_eq_0 - p_yhat_eq_1_giv_y_eq_0_z_eq_1,
            p_yhat_eq_1_giv_y_eq_0_z_eq_1 - p_yhat_eq_1_giv_y_eq_0_z_eq_0,
        ])
        if np.isnan(mu):
            mu = 1

        return mu

    def _compute_A_ub_row__split1(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=1|y=0,z=0) >= P(yhat=1|y=0,z=1)
            ```
        which is False Positive Rate Parity opt_problem 1.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        n_actions = 2
        ldf = ldf.copy()
        filt__yhat1_y0_z0 = (ldf['z'] == 0) & (ldf['y'] == 0) & (ldf['yhat'] == 1)
        filt__yhat1_y0_z1 = (ldf['z'] == 1) & (ldf['y'] == 0) & (ldf['yhat'] == 1)
        p_z0_y0 = ldf[(ldf['z'] == 0) & (ldf['y'] == 0)]['mu0'].sum() / 2
        p_z1_y0 = ldf[(ldf['z'] == 1) & (ldf['y'] == 0)]['mu0'].sum() / 2
        ldf['A_ub'] = 0.0
        ldf.loc[filt__yhat1_y0_z0, 'A_ub'] = -1 / p_z0_y0
        ldf.loc[filt__yhat1_y0_z1, 'A_ub'] =  1 / p_z1_y0

        return ldf['A_ub']

    def _compute_A_ub_row__split2(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=1|y=0,z=1) >= P(yhat=1|y=0,z=0)
            ```
        which is Equal Opportunity opt_problem 2.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        n_actions = 2
        ldf = ldf.copy()
        filt__yhat1_y0_z0 = (ldf['z'] == 0) & (ldf['y'] == 0) & (ldf['yhat'] == 1)
        filt__yhat1_y0_z1 = (ldf['z'] == 1) & (ldf['y'] == 0) & (ldf['yhat'] == 1)
        p_z0_y0 = ldf[(ldf['z'] == 0) & (ldf['y'] == 1)]['mu0'].sum() / 2
        p_z1_y0 = ldf[(ldf['z'] == 1) & (ldf['y'] == 1)]['mu0'].sum() / 2
        ldf['A_ub'] = 0.0
        ldf.loc[filt__yhat1_y0_z0, 'A_ub'] =  1 / p_z0_y0
        ldf.loc[filt__yhat1_y0_z1, 'A_ub'] = -1 / p_z1_y0

        return ldf['A_ub']

    def _construct_reward__split1(self, ldf):
        """
        Constructs the reward function for False Positive Rate Parity,
        opt_problem 1, which is when we constrain
        P(yhat=1|y=0,z=0) >= P(yhat=1|y=0,z=1), in which case the reward
        penalizes giving the Z=0 group the positive prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_y0_z0 = (ldf['z'] == 0) & (ldf['y'] == 0) & (ldf['yhat'] == 1)
        filt__yhat1_y0_z1 = (ldf['z'] == 1) & (ldf['y'] == 0) & (ldf['yhat'] == 1)
        p_y0_z0 = ldf[(ldf['y'] == 0) & (ldf['z'] == 0)]['mu0'].sum() / 2
        p_y0_z1 = ldf[(ldf['y'] == 0) & (ldf['z'] == 1)]['mu0'].sum() / 2
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_y0_z0, 'r'] = -1 / p_y0_z0
        ldf.loc[filt__yhat1_y0_z1, 'r'] =  1 / p_y0_z1
        c = -1 * ldf['r']  # Negative since maximizing not minimizing

        return c

    def _construct_reward__split2(self, ldf):
        """
        Constructs the reward function for False Positive Rate Parity,
        opt_problem 2, which is when we constrain
        P(yhat=0|y=0,z=0) <= P(yhat=0|y=0,z=1), in which case the reward
        penalizes giving the Z=1 group the positive prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_y0_z0 = (ldf['z'] == 0) & (ldf['y'] == 0) & (ldf['yhat'] == 1)
        filt__yhat1_y0_z1 = (ldf['z'] == 1) & (ldf['y'] == 0) & (ldf['yhat'] == 1)
        p_y0_z0 = ldf[(ldf['y'] == 0) & (ldf['z'] == 0)]['mu0'].sum() / 2
        p_y0_z1 = ldf[(ldf['y'] == 0) & (ldf['z'] == 1)]['mu0'].sum() / 2
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_y0_z0, 'r'] =  1 / p_y0_z0
        ldf.loc[filt__yhat1_y0_z1, 'r'] = -1 / p_y0_z1
        c = -1 * ldf['r']  # Negative since maximizing not minimizing

        return c


class TrueNegativeRateParityObjective(AbsoluteValueObjective):

    def __init__(self):
        self.name = 'TNRPar'
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_0_giv_y_eq_0_z_eq_0 = (
            ((demo['yhat'] == 0) & (demo['y'] == 0) & (demo['z'] == 0)).sum()
            / ((demo['y'] == 0) & (demo['z'] == 0)).sum()
        )
        p_yhat_eq_0_giv_y_eq_0_z_eq_1 = (
            ((demo['yhat'] == 0) & (demo['y'] == 0) & (demo['z'] == 1)).sum()
            / ((demo['y'] == 0) & (demo['z'] == 1)).sum()
        )
        mu = 1 - max([
            p_yhat_eq_0_giv_y_eq_0_z_eq_0 - p_yhat_eq_0_giv_y_eq_0_z_eq_1,
            p_yhat_eq_0_giv_y_eq_0_z_eq_1 - p_yhat_eq_0_giv_y_eq_0_z_eq_0,
        ])
        if np.isnan(mu):
            mu = 1

        return mu

    def _compute_A_ub_row__split1(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=0|y=0,z=0) >= P(yhat=0|y=0,z=1)
            ```
        which is True Negative Rate Parity opt_problem 1.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        n_actions = 2
        ldf = ldf.copy()
        filt__yhat0_y0_z0 = (ldf['z'] == 0) & (ldf['y'] == 0) & (ldf['yhat'] == 0)
        filt__yhat0_y0_z1 = (ldf['z'] == 1) & (ldf['y'] == 0) & (ldf['yhat'] == 0)
        p_z0_y0 = ldf[(ldf['z'] == 0) & (ldf['y'] == 0)]['mu0'].sum() / 2
        p_z1_y0 = ldf[(ldf['z'] == 1) & (ldf['y'] == 0)]['mu0'].sum() / 2
        ldf['A_ub'] = 0.0
        ldf.loc[filt__yhat0_y0_z0, 'A_ub'] = -1 / p_z0_y0
        ldf.loc[filt__yhat0_y0_z1, 'A_ub'] =  1 / p_z1_y0

        return ldf['A_ub']

    def _compute_A_ub_row__split2(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=0|y=0,z=1) >= P(yhat=0|y=0,z=0)
            ```
        which is True Negative Rate Parity opt_problem 2.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        n_actions = 2
        ldf = ldf.copy()
        filt__yhat0_y0_z0 = (ldf['z'] == 0) & (ldf['y'] == 0) & (ldf['yhat'] == 0)
        filt__yhat0_y0_z1 = (ldf['z'] == 1) & (ldf['y'] == 0) & (ldf['yhat'] == 0)
        p_z0_y0 = ldf[(ldf['z'] == 0) & (ldf['y'] == 0)]['mu0'].sum() / 2
        p_z1_y0 = ldf[(ldf['z'] == 1) & (ldf['y'] == 0)]['mu0'].sum() / 2
        ldf['A_ub'] = 0.0
        ldf.loc[filt__yhat0_y0_z0, 'A_ub'] =  1 / p_z0_y0
        ldf.loc[filt__yhat0_y0_z1, 'A_ub'] = -1 / p_z1_y0

        return ldf['A_ub']

    def _construct_reward__split1(self, ldf):
        """
        Constructs the reward function for True Negative Rate Parity,
        opt_problem 1, which is when we constrain
        P(yhat=0|y=0,z=0) >= P(yhat=0|y=0,z=1), in which case the reward
        penalizes giving the Z=0 group the positive prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat0_y0_z0 = (ldf['z'] == 0) & (ldf['y'] == 0) & (ldf['yhat'] == 0)
        filt__yhat0_y0_z1 = (ldf['z'] == 1) & (ldf['y'] == 0) & (ldf['yhat'] == 0)
        p_y0_z0 = ldf[(ldf['y'] == 0) & (ldf['z'] == 0)]['mu0'].sum() / 2
        p_y0_z1 = ldf[(ldf['y'] == 0) & (ldf['z'] == 1)]['mu0'].sum() / 2
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat0_y0_z0, 'r'] = -1 / p_y0_z0
        ldf.loc[filt__yhat0_y0_z1, 'r'] =  1 / p_y0_z1
        c = -1 * ldf['r']  # Negative since maximizing not minimizing

        return c

    def _construct_reward__split2(self, ldf):
        """
        Constructs the reward function for True Negative Rate Parity,
        opt_problem 2, which is when we constrain
        P(yhat=0|y=0,z=0) <= P(yhat=0|y=0,z=1), in which case the reward
        penalizes giving the Z=1 group the positive prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat0_y0_z0 = (ldf['z'] == 0) & (ldf['y'] == 0) & (ldf['yhat'] == 0)
        filt__yhat0_y0_z1 = (ldf['z'] == 1) & (ldf['y'] == 0) & (ldf['yhat'] == 0)
        p_y0_z0 = ldf[(ldf['y'] == 0) & (ldf['z'] == 0)]['mu0'].sum() / 2
        p_y0_z1 = ldf[(ldf['y'] == 0) & (ldf['z'] == 1)]['mu0'].sum() / 2
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat0_y0_z0, 'r'] =  1 / p_y0_z0
        ldf.loc[filt__yhat0_y0_z1, 'r'] = -1 / p_y0_z1
        c = -1 * ldf['r']  # Negative since maximizing not minimizing

        return c


class FalseNegativeRateParityObjective(AbsoluteValueObjective):

    def __init__(self):
        self.name = 'FNRPar'
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_0_giv_y_eq_1_z_eq_0 = (
            ((demo['yhat'] == 0) & (demo['y'] == 1) & (demo['z'] == 0)).sum()
            / ((demo['y'] == 1) & (demo['z'] == 0)).sum()
        )
        p_yhat_eq_0_giv_y_eq_1_z_eq_1 = (
            ((demo['yhat'] == 0) & (demo['y'] == 1) & (demo['z'] == 1)).sum()
            / ((demo['y'] == 1) & (demo['z'] == 1)).sum()
        )
        mu = 1 - max([
            p_yhat_eq_0_giv_y_eq_1_z_eq_0 - p_yhat_eq_0_giv_y_eq_1_z_eq_1,
            p_yhat_eq_0_giv_y_eq_1_z_eq_1 - p_yhat_eq_0_giv_y_eq_1_z_eq_0,
        ])
        if np.isnan(mu):
            mu = 1

        return mu

    def _compute_A_ub_row__split1(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=0|y=1,z=0) >= P(yhat=0|y=1,z=1)
            ```
        which is False Negative Rate Parity opt_problem 1.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        n_actions = 2
        ldf = ldf.copy()
        filt__yhat0_y1_z0 = (ldf['z'] == 0) & (ldf['y'] == 1) & (ldf['yhat'] == 0)
        filt__yhat0_y1_z1 = (ldf['z'] == 1) & (ldf['y'] == 1) & (ldf['yhat'] == 0)
        p_z0_y1 = ldf[(ldf['z'] == 0) & (ldf['y'] == 1)]['mu0'].sum() / 2
        p_z1_y1 = ldf[(ldf['z'] == 1) & (ldf['y'] == 1)]['mu0'].sum() / 2
        ldf['A_ub'] = 0.0
        ldf.loc[filt__yhat0_y1_z0, 'A_ub'] = -1 / p_z0_y1
        ldf.loc[filt__yhat0_y1_z1, 'A_ub'] =  1 / p_z1_y1

        return ldf['A_ub']

    def _compute_A_ub_row__split2(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=0|y=1,z=1) >= P(yhat=0|y=1,z=0)
            ```
        which is False Negative Rate opt_problem 2.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        n_actions = 2
        ldf = ldf.copy()
        filt__yhat0_y1_z0 = (ldf['z'] == 0) & (ldf['y'] == 1) & (ldf['yhat'] == 0)
        filt__yhat0_y1_z1 = (ldf['z'] == 1) & (ldf['y'] == 1) & (ldf['yhat'] == 0)
        p_z0_y1 = ldf[(ldf['z'] == 0) & (ldf['y'] == 1)]['mu0'].sum() / 2
        p_z1_y1 = ldf[(ldf['z'] == 1) & (ldf['y'] == 1)]['mu0'].sum() / 2
        ldf['A_ub'] = 0.0
        ldf.loc[filt__yhat0_y1_z0, 'A_ub'] =  1 / p_z0_y1
        ldf.loc[filt__yhat0_y1_z1, 'A_ub'] = -1 / p_z1_y1

        return ldf['A_ub']

    def _construct_reward__split1(self, ldf):
        """
        Constructs the reward function for False Negative Rate Parity,
        opt_problem 1, which is when we constrain
        P(yhat=0|y=1,z=0) >= P(yhat=0|y=1,z=1), in which case the reward
        penalizes giving the Z=0 group the positive prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat0_y1_z0 = (ldf['z'] == 0) & (ldf['y'] == 1) & (ldf['yhat'] == 0)
        filt__yhat0_y1_z1 = (ldf['z'] == 1) & (ldf['y'] == 1) & (ldf['yhat'] == 0)
        p_y1_z0 = ldf[(ldf['y'] == 1) & (ldf['z'] == 0)]['mu0'].sum() / 2
        p_y1_z1 = ldf[(ldf['y'] == 1) & (ldf['z'] == 1)]['mu0'].sum() / 2
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat0_y1_z0, 'r'] = -1 / p_y1_z0
        ldf.loc[filt__yhat0_y1_z1, 'r'] =  1 / p_y1_z1
        c = -1 * ldf['r']  # Negative since maximizing not minimizing

        return c

    def _construct_reward__split2(self, ldf):
        """
        Constructs the reward function for False Negative Rate Parity,
        opt_problem 2, which is when we constrain
        P(yhat=0|y=1,z=0) <= P(yhat=0|y=1,z=1), in which case the reward
        penalizes giving the Z=1 group the positive prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat0_y1_z0 = (ldf['z'] == 0) & (ldf['y'] == 1) & (ldf['yhat'] == 0)
        filt__yhat0_y1_z1 = (ldf['z'] == 1) & (ldf['y'] == 1) & (ldf['yhat'] == 0)
        p_y1_z0 = ldf[(ldf['y'] == 1) & (ldf['z'] == 0)]['mu0'].sum() / 2
        p_y1_z1 = ldf[(ldf['y'] == 1) & (ldf['z'] == 1)]['mu0'].sum() / 2
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat0_y1_z0, 'r'] =  1 / p_y1_z0
        ldf.loc[filt__yhat0_y1_z1, 'r'] = -1 / p_y1_z1
        c = -1 * ldf['r']  # Negative since maximizing not minimizing

        return c


class PredictiveParityObjective(Objective):
    """
    Not intended for optimization, just feature expectations. Can't optimize
    this with LP, since it's not a linear measure.
    """

    def __init__(self):
        self.name = 'PredPar'
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_y_eq_1_giv_yhat_eq_1_z_eq_0 = (
            ((demo['yhat'] == 1) & (demo['y'] == 1) & (demo['z'] == 0)).sum()
            / ((demo['yhat'] == 1) & (demo['z'] == 0)).sum()
        )
        p_y_eq_1_giv_yhat_eq_1_z_eq_1 = (
            ((demo['yhat'] == 1) & (demo['y'] == 1) & (demo['z'] == 1)).sum()
            / ((demo['yhat'] == 1) & (demo['z'] == 1)).sum()
        )
        mu = 1 - max([
            p_y_eq_1_giv_yhat_eq_1_z_eq_0 - p_y_eq_1_giv_yhat_eq_1_z_eq_1,
            p_y_eq_1_giv_yhat_eq_1_z_eq_1 - p_y_eq_1_giv_yhat_eq_1_z_eq_0,
        ])
        if np.isnan(mu):
            mu = 1

        return mu


class NegativePredictiveParityObjective(Objective):
    """
    Not intended for optimization, just feature expectations. Can't optimize
    this with LP, since it's not a linear measure.
    """

    def __init__(self):
        self.name = 'NegPredPar'
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_y_eq_0_giv_yhat_eq_0_z_eq_0 = (
            ((demo['yhat'] == 0) & (demo['y'] == 0) & (demo['z'] == 0)).sum()
            / ((demo['yhat'] == 0) & (demo['z'] == 0)).sum()
        )
        p_y_eq_0_giv_yhat_eq_0_z_eq_1 = (
            ((demo['yhat'] == 0) & (demo['y'] == 0) & (demo['z'] == 1)).sum()
            / ((demo['yhat'] == 0) & (demo['z'] == 1)).sum()
        )
        mu = 1 - max([
            p_y_eq_0_giv_yhat_eq_0_z_eq_0 - p_y_eq_0_giv_yhat_eq_0_z_eq_1,
            p_y_eq_0_giv_yhat_eq_0_z_eq_1 - p_y_eq_0_giv_yhat_eq_0_z_eq_0,
        ])
        if np.isnan(mu):
            mu = 1

        return mu

class GroupPositiveRateZ0Objective(LinearObjective):

    def __init__(self, z=0):
        self.z = z
        self.name = f"PR_Z{self.z}"
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_1_giv_z_eq_z = (
            (
                (demo['yhat'] == 1) & (demo['z'] == self.z)
            ).sum() / (demo['z'] == self.z).sum()
        )

        mu = p_yhat_eq_1_giv_z_eq_z

        if np.isnan(mu):
            mu = 1

        return mu

    def _construct_reward(self, ldf):
        """
        Constructs the reward function component for this objective.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        ------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_giv_z = (
            (ldf['z'] == self.z) & (ldf['yhat'] == 1)
        )
        # Divide by 2 since ldf has two duplicate rows per state
        p_z = ldf[ldf['z'] == self.z]['mu0'].sum() / 2
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_giv_z, 'r'] = 1 / p_z
        c = -1 * ldf['r']  # Negative since maximizing not minimizing

        # Check c sums to 1
        assert (c.sum() + 1) < 1e-8

        return c


class GroupPositiveRateZ1Objective(GroupPositiveRateZ0Objective):

    def __init__(self):
        super().__init__(z=1)


class GroupNegativeRateZ0Objective(LinearObjective):

    def __init__(self, z=0):
        self.z = z
        self.name = f"NR_Z{self.z}"
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_0_giv_z_eq_z = (
            (
                (demo['yhat'] == 0) & (demo['z'] == self.z)
            ).sum() / (demo['z'] == self.z).sum()
        )

        mu = p_yhat_eq_0_giv_z_eq_z

        if np.isnan(mu):
            mu = 1

        return mu

    def _construct_reward(self, ldf):
        """
        Constructs the reward function component for this objective.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        ------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat0_giv_z = (
            (ldf['z'] == self.z) & (ldf['yhat'] == 0)
        )
        # Divide by 2 since ldf has two duplicate rows per state
        p_z = ldf[ldf['z'] == self.z]['mu0'].sum() / 2
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat0_giv_z, 'r'] = 1 / p_z
        c = -1 * ldf['r']  # Negative since maximizing not minimizing

        # Check c sums to 1
        assert (c.sum() + 1) < 1e-8

        return c


class GroupNegativeRateZ1Objective(GroupNegativeRateZ0Objective):

    def __init__(self):
        super().__init__(z=1)

class GroupTruePositiveRateZ0Objective(LinearObjective):

    def __init__(self, z=0):
        self.z = z
        self.name = f"TPR_Z{self.z}"
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_1_giv_y_eq_1_z_eq_z = (
            (
                (demo['yhat'] == 1) & (demo['y'] == 1) & (demo['z'] == self.z)
            ).sum() / ((demo['y'] == 1) & (demo['z'] == self.z)).sum()
        )

        mu = p_yhat_eq_1_giv_y_eq_1_z_eq_z

        if np.isnan(mu):
            mu = 1

        return mu

    def _construct_reward(self, ldf):
        """
        Constructs the reward function component for this objective.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        ------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_giv_y1_z = (
            (ldf['z'] == self.z) & (ldf['y'] == 1) & (ldf['yhat'] == 1)
        )
        # Divide by 2 since ldf has two duplicate rows per state
        p_y1_z = ldf[(ldf['y'] == 1) & (ldf['z'] == self.z)]['mu0'].sum() / 2
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_giv_y1_z, 'r'] = 1 / p_y1_z
        c = -1 * ldf['r']  # Negative since maximizing not minimizing

        # Check c sums to 1
        assert (c.sum() + 1) < 1e-8

        return c


class GroupTruePositiveRateZ1Objective(GroupTruePositiveRateZ0Objective):

    def __init__(self):
        super().__init__(z=1)


class GroupTrueNegativeRateZ0Objective(LinearObjective):

    def __init__(self, z=0):
        self.z = z
        self.name = f"TNR_Z{self.z}"
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_0_giv_y_eq_0_z_eq_z = (
            (
                (demo['yhat'] == 0) & (demo['y'] == 0) & (demo['z'] == self.z)
            ).sum() / ((demo['y'] == 0) & (demo['z'] == self.z)).sum()
        )

        mu = p_yhat_eq_0_giv_y_eq_0_z_eq_z

        if np.isnan(mu):
            mu = 1

        return mu

    def _construct_reward(self, ldf):
        """
        Constructs the reward function component for this objective.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        ------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat0_giv_y0_z = (
            (ldf['z'] == self.z) & (ldf['y'] == 0) & (ldf['yhat'] == 0)
        )
        p_y0_z = ldf[(ldf['y'] == 0) & (ldf['z'] == self.z)]['mu0'].sum() / 2
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat0_giv_y0_z, 'r'] = 1 / p_y0_z
        c = -1 * ldf['r']  # Negative since maximizing not minimizing

        return c


class GroupTrueNegativeRateZ1Objective(GroupTrueNegativeRateZ0Objective):

    def __init__(self):
        super().__init__(z=1)


class GroupFalsePositiveRateZ0Objective(LinearObjective):

    def __init__(self, z=0):
        self.z = z
        self.name = f"FPR_Z{self.z}"
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_1_giv_y_eq_0_z_eq_z = (
            (
                (demo['yhat'] == 1) & (demo['y'] == 0) & (demo['z'] == self.z)
            ).sum() / ((demo['y'] == 0) & (demo['z'] == self.z)).sum()
        )

        mu = p_yhat_eq_1_giv_y_eq_0_z_eq_z

        # Absence should imply 0
        if np.isnan(mu):
            mu = 1.

        return mu

    def _construct_reward(self, ldf):
        """
        Constructs the reward function component for this objective.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        ------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_giv_y0_z = (
            (ldf['z'] == self.z) & (ldf['y'] == 0) & (ldf['yhat'] == 1)
        )
        p_y0_z = ldf[(ldf['y'] == 0) & (ldf['z'] == self.z)]['mu0'].sum() / 2
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_giv_y0_z, 'r'] = 1 / p_y0_z
        c = -1 * ldf['r']  # Negative since maximizing not minimizing

        return c


class GroupFalsePositiveRateZ1Objective(GroupFalsePositiveRateZ0Objective):

    def __init__(self):
        super().__init__(z=1)


class GroupFalseNegativeRateZ0Objective(LinearObjective):

    def __init__(self, z=0):
        self.z = z
        self.name = f"FNR_Z{self.z}"
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_1_giv_y_eq_0_z_eq_z = (
            (
                (demo['yhat'] == 1) & (demo['y'] == 0) & (demo['z'] == self.z)
            ).sum() / ((demo['y'] == 0) & (demo['z'] == self.z)).sum()
        )

        mu = p_yhat_eq_1_giv_y_eq_0_z_eq_z

        # Absence should imply 1
        if np.isnan(mu):
            mu = 1.

        return mu

    def _construct_reward(self, ldf):
        """
        Constructs the reward function component for this objective.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        ------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_giv_y0_z = (
            (ldf['z'] == self.z) & (ldf['y'] == 0) & (ldf['yhat'] == 1)
        )
        p_y0_z = ldf[(ldf['y'] == 0) & (ldf['z'] == self.z)]['mu0'].sum() / 2
        ldf['r'] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_giv_y0_z, 'r'] = 1 / p_y0_z
        c = -1 * ldf['r']  # Negative since maximizing not minimizing

        return c


class GroupFalseNegativeRateZ1Objective(GroupFalseNegativeRateZ0Objective):

    def __init__(self):
        super().__init__(z=1)


class ObjectiveSet():
    """
    The set of all objectives that make up the space of possible objectives in
    the reward function. An obj_set gets paired with a set of reward
    weights to form the full reward function.

    Parameters
    ----------
    objectives : array-like<Objective>
        Array of all objectives.

    Attributes
    ----------
    opt_problems_ : list<OptimizationProblem>
        Optimization problems.
    """

    def __init__(self, objectives):
        self.objectives = objectives
        self.opt_problems_ = None

    def compute_demo_feature_exp(self, demo):
        """
        Computes the feature expectations for a set of demonstrations.
        Does NOT need to be fit for this method to work.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns are
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets

        Returns
        -------
        mu : array<float>, len(len(self.objectives))
            The feature expectations.
        """
        mu = [obj.compute_feat_exp(demo) for obj in self.objectives]
        return mu

    def fit(self, reward_weights, ldf, A_eq, b_eq):
        """
        Constructs all optimization problems for the given objectives,
        reward weights, and MDP (represented by ldf, A_eq, b_eq).

        Parameters
        ----------
        reward_weights : dict<str, float>
            Keys are objective identifiers. Values are their respective reward
            weights.
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.
        A_eq : 2-D array
            The equality constraint matrix. Each row of ``A_eq`` specifies the
            coefficients of a linear equality constraint on ``x``.
        b_eq : 1-D array
            The equality constraint vector. Each element of ``A_eq @ x`` must
            equal the corresponding element of ``b_eq``.

        Sets
        ----
        opt_problems_ : list<OptimizationProblem>
            Optimization problems.

        Returns
        -------
        self
        """
        self.objectives = [obj.fit(ldf) for obj in self.objectives]

        abs_val_splits = []
        linear_splits = []
        abs_val_split_generator = []
        counter = 0
        for obj in self.objectives:
            if obj.n_splits == 1:
                linear_splits += obj.to_splits()
            if obj.n_splits == 2:
                abs_val_splits += obj.to_splits()
                abs_val_split_generator.append([counter, counter+1])
                counter += 2


        # abs_val_splits_perms = list(itertools.product(*[[0,1], [2,3]]))
        # >>> [(0, 2), (0, 3), (1, 2), (1, 3)]
        abs_val_splits_perms = list(itertools.product(*abs_val_split_generator))
        opt_problems = []
        for split_indexes in abs_val_splits_perms:
            _abs_val_splits = [abs_val_splits[idx] for idx in split_indexes]
            all_splits = linear_splits + _abs_val_splits
            name = ','.join([split.name for split in all_splits])
            A_ub = np.array([split.A_ub for split in _abs_val_splits], dtype=float)
            b_ub = np.array([split.b_ub for split in _abs_val_splits], dtype=float)
            c = reward_weights[all_splits[0].parent.name] * all_splits[0].c
            for split_i, split in enumerate(all_splits[1:]):
                c += reward_weights[split.parent.name] * split.c

            opt_problem = OptimizationProblem(
                name=name,
                A_eq=A_eq,
                b_eq=b_eq,
                A_ub=A_ub,
                b_ub=b_ub,
                c=c,
            )

            opt_problems.append(opt_problem)

        self.opt_problems_ = opt_problems

        return self

    def reset(self):
        """
        Re-initializes each of the objectives and self.opt_problems_.

        Parameters
        ----------
        N/A

        Unsets
        ------
        opt_problems_

        Returns
        -------
        None
        """
        for obj in self.objectives:
            obj.__init__()

        self.opt_problems_ = None
