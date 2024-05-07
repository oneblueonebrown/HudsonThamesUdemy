"""Class for the implementation of Kinetic Component Analysis"""

from typing import Optional
import numpy as np
import pandas as pd
import pykalman as pykf


class KCA:
    """Kinetic Component Analysis is a framework to obtain signals from noisy
    measurements of a given stochastic process by applying a Kalman filter on a Taylor
    expansion.

    As in the calculation of the well-known momentum signals from price data, Kinetic
    Component Analysis aims to provide an indication for *inertia* in price dynamics
    by decomposing the signals in three (non-observable) variables linked to position,
    velocity and acceleration.

    A thorough analysis can be found at
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2422183
    """

    def __init__(
        self,
        noisy_price_data: pd.Series
    ) -> None:

        self.raw_data = noisy_price_data

        # It is worth noting that if we are dealing with datetime index,
        # converting it to float will replace datetime objects with
        # equivalent nanosecond data
        self.time_ticks = noisy_price_data.index.values.astype(float)
        self.noisy_signal = noisy_price_data.values
        self.tick_size = self._set_tick_size()

    def _set_tick_size(self) -> float:
        """Calculates the fixed frequency of the interval
        between samples.

        Returns:
            float: value of the fixed frequency.
        """

        n_obs = self.time_ticks.shape[0]
        time_delta = self.time_ticks[-1] - self.time_ticks[0]
        time_step_h = time_delta / n_obs

        # Normalizing the time step h when working with datetime indexes
        if isinstance(self.raw_data.index, pd.DatetimeIndex):
            time_step_h /= self.time_ticks.max()

        return time_step_h

    def _components_std(self, x_covar: np.ndarray) -> np.ndarray:
        """Calculates the standard deviation of the process' state components

        Args:
            x_covar (np.ndarray): covariance matrix of the state vector.

        Returns:
            np.ndarray: standard deviation of the components of the state vector.
        """

        x_std = (x_covar[:, 0, 0]**.5).reshape(-1, 1)
        for i in range(1, x_covar.shape[1]):
            x_std_ = x_covar[:, i, i]**.5
            x_std = np.append(x_std, x_std_.reshape(-1, 1), axis=1)
        return x_std

    def _fit_kca(self, q_mat_seed: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Wraps the implementation of a Kalman filter using the pykalman package's
        KalmanFilter class, leveraging its ```smooth``` and ```em``` methods for
        optimizing the parameters of the Kalman filter over the noisy measurement
        data input.

        More details about the pykalman package can be found at https://pykalman.github.io/

        Args:
            q_mat_seed (float): _description_

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: _description_
        """

        # TODO: extend functionality to include higher degrees for the Taylor series
        # since the following implementation works for n = 2 only

        #1) Set up matrices A, H and a seed for Q
        mat_a = np.array(
            [[1, self.tick_size, (self.tick_size / 2) ** 2],
            [0 , 1, self.tick_size],
            [0, 0, 1]]
        )

        mat_q = q_mat_seed * np.eye(mat_a.shape[0])

        #2) Apply the filter
        kf = pykf.KalmanFilter(transition_matrices=mat_a,transition_covariance=mat_q)

        #3) EM estimates
        kf = kf.em(self.noisy_signal)

        #4) Smooth
        x_mean, x_covar = kf.smooth(self.noisy_signal)

        return x_mean, x_covar

    def get_kca_components(
        self, q_seed: float, include_std: Optional[bool]=False
    ) -> pd.DataFrame:
        """Returns the Kalman Filter result for the noisy measurement data given
        a seed for the $Q$ matrix.

        Args:
            q_seed (float): scalar that multiplies the Identity matrix
            include_std (Optional[bool], optional): Flag to add the standard
            deviation of the estimates. Defaults to False.

        Returns:
            pd.DataFrame: _description_
        """
        x_mean, x_covar = self._fit_kca(q_mat_seed=q_seed)
        kf_result = pd.DataFrame(
            index=self.raw_data.index,
            data=x_mean,
            columns=["position", "velocity", "acceleration"]
        )
        if include_std:
            x_std = self._components_std(x_covar=x_covar)
            kf_result[[col + "_std" for col in kf_result.columns]] = x_std

        return kf_result
