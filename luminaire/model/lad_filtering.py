from luminaire.model.base_model import BaseModel, BaseModelObject, BaseModelHyperParams
from luminaire.exploration.data_exploration import DataExploration
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class LADFilteringHyperParams(BaseModelHyperParams):
    """
    Exception class for Luminaire filtering anomaly detection model.

    :param bool is_log_transformed: A flag to specify whether to take a log transform of the input data. If the data
        contain negatives, is_log_transformed is ignored even though it is set to True.
    :type is_log_transformed: bool, optional
    """
    def __init__(self,
                 is_log_transformed=True):

        super(LADFilteringHyperParams, self).__init__(
            model_name="LADFilteringModel",
            is_log_transformed=is_log_transformed,
        )


class LADFilteringModelError(Exception):
    """
    Exception class for Luminaire filtering anomaly detection model.

    """

    def __init__(self, message):
        message = f'State Space model failed! Error: {message}'

        super(LADFilteringModelError, self).__init__(message)


class LADFilteringModel(BaseModel):
    """
    A Markovian state space model. This model detects anomaly based on the residual process obtained
    through Kalman Filter based model estimation.

    :param dict hyper_params: Hyper parameters for Luminaire structural modeling.
        See :class:`luminaire.optimization.hyperparameter_optimization.HyperparameterOptimization` for detailed information.
    :param str freq: The frequency of the time-series. A `Pandas offset`_ such as 'D', 'H', or 'M'.
    :param min_ts_length: The minimum required length of the time series for training.
    :type min_ts_length: int, optional
    :param max_ts_length: The maximum required length of the time series for training.
    :type max_ts_length: int, optional

    >>> hyper = {"is_log_transformed": 1}
    lad_filtering_model = LADFilteringModel(hyper_params=hyper, freq='D')

    >>> lad_filtering_model
    <luminaire.model.filtering.LADFilteringModel object at 0x103efe320>
    """

    __version__ = "2.0"

    _target_metric = 'raw'
    _imputed_metric = 'interpolated'
    _sig_level = 0.10
    _sig_level_extreme = 0.001

    max_scoring_length_dict = {
        'H': 48,
        'D': 10,
        'W': 8, 'W-SUN': 8, 'W-MON': 8, 'W-TUE': 8, 'W-WED': 8, 'W-THU': 8, 'W-FRI': 8, 'W-SAT': 8,
        'M': 24, 'MS': 24,
    }

    def __init__(self,
                 hyper_params: LADFilteringHyperParams().params,
                 freq,
                 min_ts_length=None,
                 max_ts_length=None,
                 **kwargs):

        self.hyper_params = hyper_params

        super(LADFilteringModel, self).__init__(freq=freq,
                                                min_ts_length=min_ts_length,
                                                max_ts_length=max_ts_length, **hyper_params, **kwargs)

    @classmethod
    def _prediction_summary(cls, state_mean, state_covariance, observation_covariance, transition_covariance,
                            observation_matrix, transition_matrix):
        """
        This function computes the prediction summary given the current state of the time series.

        :param float state_mean: State mean given the latest time point.
        :param numpy.array state_covariance: State covariance given the latest time point.
        :param numpy.array observation_covariance: Covariance of the observed time series.
        :param numpy.array transition_covariance: State prediction covariance given the current time point.
        :param numpy.array observation_matrix: Matrix mapping the hidden states to the observed states.
        :param numpy.array transition_matrix: Matrix mapping the current hidden state to the next hidden state.
        :return: A tuple containing prior prediction for the next state, prediction covariance and the
        kalman gain
        """
        import numpy as np

        try:

            ### Computing the parameters of the Gaussian residual process ###
            ### Reference for the detailed algorithm: (Soule et. al., Combining filtering and statistical methods for
            ### anomaly detection, Proceedings of the 5th ACM SIGCOMM, 2005)
            pred_covariance = np.matmul(np.matmul(np.array(transition_matrix), state_covariance),
                                        np.array(transition_matrix).T) + transition_covariance

            # Calculating the Kalman Gain (Equation 6 from the paper)
            kalman_term1 = np.matmul(pred_covariance, observation_matrix.T)
            kalman_term2 = np.matmul(np.matmul(observation_matrix, pred_covariance),
                                     observation_matrix.T) + observation_covariance
            kalman_gain = np.matmul(kalman_term1, np.linalg.inv(kalman_term2))

            # Prior prediction based the Kalman Filter of the following state given the current state
            prior_pred = transition_matrix[0][0] * state_mean

        except Exception as e:
            # If ARIMA fails, then we use the arima_error flag that is going to execute the alternate EWMA based model
            raise LADFilteringModelError(message=str(e))

        return prior_pred, pred_covariance, kalman_gain

    @classmethod
    def _training(self, data, **kwargs):
        """
        This function implements Kalman filter based estimation algorithm over a Markovian State Space model and
        analyzes the residual process of the model with respect to a Gaussian process to perform anomaly detection
        :param pandas.DataFrame data: Input time seires to analyze for anomaly
        :param float sig_level: Significance level to be considered for anomaly detection based on the Gaussian process
        :return: A tuple containing a flag whether the datapoint on the given date is an anomnaly, the prediction and
        the standard error of prediction
        """

        import numpy as np
        from pykalman import KalmanFilter
        from numpy.linalg import LinAlgError

        if data is None:
            raise ValueError('Not enough data to train due to recent change point')

        data = data[self._imputed_metric]

        last_data_points = data[-2:].values.tolist()

        try:
            data_dim = 1
            transition_matrix = [[1]]

            de_obj = DataExploration()
            endog, diff_order, actual_previous_per_diff = de_obj._stationarizer(data)

            kf = KalmanFilter(transition_matrices=transition_matrix, initial_state_mean=np.zeros(data_dim),
                              n_dim_obs=data_dim)

            # Obtaining the hidden states and their covariance based on the Kalman Filter algorithm
            filtered_state_means, filtered_state_covariance = kf.em(endog).filter(endog)

            # Obtaining the observation matirx, transition covariance and the observation covariance
            observation_matrix = kf.observation_matrices
            transition_covariance = kf.transition_covariance
            observation_covariance = kf.observation_covariance

            prior_pred, pred_covariance, kalman_gain \
                = self._prediction_summary(state_mean=filtered_state_means[:, 0][-1],
                                           state_covariance=filtered_state_covariance[-1, :, :],
                                           observation_covariance=observation_covariance,
                                           transition_covariance=transition_covariance,
                                           observation_matrix=observation_matrix,
                                           transition_matrix=transition_matrix)

            result = {'model': kf,
                      'state_mean': float(filtered_state_means[:, 0][-1]),
                      'state_covariance': filtered_state_covariance[-1, :, :].tolist(),
                      'transition_matrix': transition_matrix,
                      'prior_pred': float(prior_pred),
                      'pred_covariance': pred_covariance.tolist(),
                      'kalman_gain': kalman_gain.tolist(),
                      'diff_order': diff_order,
                      'last_data_points': last_data_points}

        except (LinAlgError, ValueError, LADFilteringModelError) as e:
            result = {'ErrorMessage': str(e)}

        return result


    def train(self, data, **kwargs):
        """
        This function trains a filtering LAD model for a given time series.

        :param pandas.DataFrame data: Input time series data
        :return: The success flag, model date and a trained lad filtering object
        :rtype: tuple[bool, str, LADFilteringModel object]

        >>> data
                       raw interpolated
        2020-01-01  1326.0       1326.0
        2020-01-02  1552.0       1552.0
        2020-01-03  1432.0       1432.0
        2020-01-04  1470.0       1470.0
        2020-01-05  1565.0       1565.0
        ...            ...          ...
        2020-06-03  1934.0       1934.0
        2020-06-04  1873.0       1873.0
        2020-06-05  1674.0       1674.0
        2020-06-06  1747.0       1747.0
        2020-06-07  1782.0       1782.0
        >>> hyper = {"is_log_transformed": 1}
        >>> de_obj = DataExploration(freq='D', is_log_transformed=1, fill_rate=0.95)
        >>> data, pre_prc = de_obj.profile(data)
        >>> pre_prc
        {'success': True, 'trend_change_list': ['2020-04-01 00:00:00'], 'change_point_list': ['2020-03-16 00:00:00'],
        'is_log_transformed': 1, 'min_ts_mean': None, 'ts_start': '2020-01-01 00:00:00',
        'ts_end': '2020-06-07 00:00:00'}
        >>> lad_filter_obj = LADFilteringModel(hyper_params=hyper, freq='D')
        >>> model = lad_filter_obj.train(data=data, **pre_prc)

        >>> model
        (True, '2020-06-07 00:00:00', <luminaire.model.lad_filtering.LADFilteringModel object at 0x11b6c4f60>)
        """

        result = self._training(data=data, **kwargs)

        self.hyper_params['is_log_transformed'] = kwargs['is_log_transformed']
        result['training_end_date'] = kwargs['ts_end']
        result['freq'] = self._params['freq']

        success = False if 'ErrorMessage' in result else True

        return success, kwargs['ts_end'], LADFilteringModel(hyper_params=self.hyper_params, **result)

    @classmethod
    def _scoring(cls, raw_actual=None, synthetic_actual=None, model=None, state_mean=None, training_end=None,
                 pred_date=None, state_covariance=None, transition_matrix=None, prior_pred=None, pred_covariance=None,
                 diff_order=None, kalman_gain=None, interpolated_actual_previous=None, is_log_transformed=None,
                 freq=None):
        """
        This function performs scoring using the state space model object

        :param float raw_actual: Observed time series value
        :param float synthetic_actual: Synthetic time series value
        :param python object model: State space model object
        :param numpy.array state_mean: Hidden states for the given time series
        :param str training_end: Last time series timestamp
        :param str pred_date: Prediction date
        :param numpy.array state_covariance: Covariance matrix for the hidden states
        :param numpy.array transition_matrix: Matrix for the hidden state transition
        :param float prior_pred: Prior prediction for the current state
        :param numpy.array pred_covariance: Prediction covariance
        :param int diff_order: Order of differencing for the nonstationarity property of the given time series
        :param numpy.array kalman_gain: Kalman gain
        :param list interpolated_actual_previous: Padding from latest time series interpolated values for prediction
        :param bool is_log_transformed: Flag for log transformation
        :param str freq: Frequency of the observed time series
        :return: Model result and the model object
        :rtype: tuple[dict, dict]
        """

        import numpy as np
        import scipy.stats as st
        from numpy.linalg import LinAlgError
        is_anomaly = False
        float_min = 1e-10

        observation_matrix = model.observation_matrices
        transition_covariance = model.transition_covariance
        observation_covariance = model.observation_covariance

        try:
            exact_freq = "1" + freq if not any(char.isdigit() for char in freq) else freq
            forecast_ndays = int((pred_date - pd.Timestamp(training_end)) // pd.Timedelta(exact_freq))
            max_scoring_length = cls.max_scoring_length_dict.get(freq)
            if forecast_ndays > max_scoring_length:
                raise ValueError('Current trained model object expired')
            model_freshness = forecast_ndays / float(max_scoring_length)

            if not synthetic_actual:
                if is_log_transformed:
                    interpolated_actual = 0 if (raw_actual is None or raw_actual <= 0) else np.log(raw_actual + 1)
                else:
                    interpolated_actual = 0 if raw_actual is None else raw_actual
                interpolated_last = interpolated_actual
            else:
                interpolated_actual = np.log(synthetic_actual + 1) if is_log_transformed else synthetic_actual
                interpolated_last = np.log(raw_actual+1) if is_log_transformed else raw_actual

            last_data_points = [interpolated_actual_previous[-1], interpolated_last]

            if diff_order:
                actual_previous_per_diff = [interpolated_actual_previous[-1]] \
                    if diff_order == 1 else [interpolated_actual_previous[-1], np.diff(interpolated_actual_previous)[0]]
                seq_tail = interpolated_actual_previous + [interpolated_actual]
                interpolated_actual = np.diff(seq_tail, 2)[-1]

            post_pred = prior_pred + kalman_gain[0][0] * (interpolated_actual - (observation_matrix[0][0] * prior_pred))

            pred_error_t = post_pred - prior_pred
            gproc_variance = np.matmul(np.matmul(pred_covariance, observation_matrix.T), kalman_gain.T)

            # Computing the estimation error (Equation 9 from the paper)
            gproc = np.matmul(kalman_gain, observation_matrix)
            gproc = np.matmul(gproc, pred_covariance)
            gproc = np.matmul(gproc, np.linalg.inv(gproc_variance))
            gproc = (-1) * gproc[0][0] * pred_error_t

            # Anomaly detection based on the confidence interval of the Gaussian process (Section 3.2 of the paper)
            zscore = gproc / max(float(np.sqrt(state_covariance)), float_min)
            anomaly_probability = float((2 * st.norm(0, 1).cdf(abs(zscore))) - 1) if raw_actual is not None else 1
            down_anomaly_probability = float(1 - st.norm(0, 1).cdf(-zscore)) if raw_actual is not None else 1
            up_anomaly_probability = float(st.norm(0, 1).cdf(-zscore)) if raw_actual is not None else 1
            is_anomaly = anomaly_probability > 1 - cls._sig_level
            is_anomaly_extreme = anomaly_probability > 1 - cls._sig_level_extreme
            prediction = post_pred if not diff_order else (post_pred + np.sum(actual_previous_per_diff))
            prediction_std_err = np.sqrt(pred_covariance[0, 0])

            state_mean, state_covariance = model.filter_update(filtered_state_mean=np.array(state_mean),
                                                               filtered_state_covariance=state_covariance,
                                                               observation=interpolated_actual)
            state_mean = np.array(state_mean)[0][0]

            prior_pred, pred_covariance, kalman_gain \
                = cls._prediction_summary(state_mean=state_mean,
                                          state_covariance=state_covariance,
                                          observation_covariance=observation_covariance,
                                          transition_covariance=transition_covariance,
                                          observation_matrix=observation_matrix,
                                          transition_matrix=transition_matrix)

            # Lower and the upper confidence interval
            if is_log_transformed:
                transformed_back_prediction = np.exp(prediction + ((prediction_std_err ** 2) / 2.0)) - 1

                transformed_back_std_err = np.sqrt((np.exp(prediction_std_err ** 2) - 1) *
                                                   (np.exp((2 * prediction) + (prediction_std_err ** 2))))

                transformed_back_interpolated_actual = float(np.exp(interpolated_actual) - 1)

                result = {'Success': True,
                          'AdjustedActual': float(transformed_back_interpolated_actual),
                          'ConfLevel': float(1.0 - cls._sig_level) * 100,
                          'Prediction': float(transformed_back_prediction),
                          'PredStdErr': float(transformed_back_std_err),
                          'IsAnomaly': is_anomaly,
                          'IsAnomalyExtreme': is_anomaly_extreme,
                          'AnomalyProbability': 1 if raw_actual is None else anomaly_probability,
                          'DownAnomalyProbability': 1 if raw_actual is None else down_anomaly_probability,
                          'UpAnomalyProbability': 1 if raw_actual is None else up_anomaly_probability,
                          'NonStationarityDiffOrder': diff_order,
                          'ModelFreshness': model_freshness
                          }
            else:
                result = {'Success': True,
                          'AdjustedActual': float(interpolated_actual),
                          'ConfLevel': float(1.0 - cls._sig_level) * 100,
                          'Prediction': float(prediction),
                          'PredStdErr': float(prediction_std_err),
                          'IsAnomaly': is_anomaly,
                          'IsAnomalyExtreme': is_anomaly_extreme,
                          'AnomalyProbability': 1 if raw_actual is None else anomaly_probability,
                          'DownAnomalyProbability': 1 if raw_actual is None else down_anomaly_probability,
                          'UpAnomalyProbability': 1 if raw_actual is None else up_anomaly_probability,
                          'NonStationarityDiffOrder': diff_order,
                          'ModelFreshness': model_freshness
                          }
            model = {'model': model,
                     'state_mean': float(state_mean),
                     'state_covariance': state_covariance.tolist(),
                     'transition_matrix': transition_matrix,
                     'prior_pred': float(prior_pred),
                     'pred_covariance': pred_covariance.tolist(),
                     'kalman_gain': kalman_gain.tolist(),
                     'diff_order': diff_order,
                     'last_data_points': last_data_points,
                     'training_end_date': training_end,
                     'freq': freq}

        except (LinAlgError, ValueError, LADFilteringModelError) as e:
            result = {'Success': False,
                      'ErrorMessage': str(e)}
            model = None

        return result, model

    def score(self, observed_value, pred_date, synthetic_actual=None, **kwargs):
        """
        This function scores a value observed at a data date given a trained LAD filtering model object.

        :param float observed_value: Observed time series value on the prediction date.
        :param str pred_date: Prediction date. Needs to be in yyyy-mm-dd or yyyy-mm-dd hh:mm:ss format.
        :param float synthetic_actual: Synthetic time series value. This is an artificial value used to optimize
            classification accuracy in Luminaire hyperparameter optimization.
        :type synthetic_actual: float, optional
        :return: Model results and LAD filtering model object
        :rtype: tuple[dict, LADFilteringlModel object]

        >>> model
        <luminaire.model.lad_filtering.LADFilteringModel object at 0x11f0b2b38>
        >>> model._params['training_end_date']
        '2020-06-07 00:00:00'

        >>> model.score(2000 ,'2020-06-08')
        ({'Success': True, 'AdjustedActual': 0.10110881711268949, 'ConfLevel': 90.0, 'Prediction': 1934.153554885343,
        'PredStdErr': 212.4399633739204, 'IsAnomaly': False, 'IsAnomalyExtreme': False,
        'AnomalyProbability': 0.4244056403219776, 'DownAnomalyProbability': 0.2877971798390112,
        'UpAnomalyProbability': 0.7122028201609888, 'NonStationarityDiffOrder': 2, 'ModelFreshness': 0.1},
        <luminaire.model.lad_filtering.LADFilteringModel object at 0x11f3c0860>)
        """

        import pandas as pd
        import numpy as np

        pred_date = pd.Timestamp(pred_date)

        result, model = self._scoring(raw_actual=observed_value, synthetic_actual=synthetic_actual,
                                      model=self._params['model'],
                                      state_mean=np.array(self._params['state_mean']),
                                      state_covariance=np.array(self._params['state_covariance']),
                                      transition_matrix=self._params['transition_matrix'],
                                      pred_date=pred_date,
                                      is_log_transformed=self._params['is_log_transformed'],
                                      diff_order=self._params['diff_order'],
                                      training_end=self._params['training_end_date'],
                                      prior_pred=self._params['prior_pred'],
                                      kalman_gain=np.array(self._params['kalman_gain']),
                                      pred_covariance=np.array(self._params['pred_covariance']),
                                      interpolated_actual_previous=self._params['last_data_points'],
                                      freq=self._params['freq'])

        model_obj = LADFilteringModel(hyper_params=self.hyper_params, **model) if model else None

        return result, model_obj