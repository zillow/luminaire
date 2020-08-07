from luminaire.model.base_model import BaseModel, BaseModelObject, BaseModelHyperParams
from luminaire.exploration.data_exploration import DataExploration
from luminaire.model.model_utils import LADHolidays
from typing import Dict, Tuple
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class LADStructuralHyperParams(BaseModelHyperParams):
    """
    Exception class for Luminaire structural anomaly detection model.

    :param bool include_holidays_exog: whether to include holidays as exogenous variables in the regression. Holidays
        are defined in :class:`~model.model_utils.LADHolidays`
    :type include_holidays_exog: bool, optional
    :param int p: Order for the AR component of the model.
    :type p: int, optional
    :param int q: Order for the MA component of the model.
    :type q: int, optional
    :param bool is_log_transformed: A flag to specify whether to take a log transform of the input data. If the data
        contain negatives, is_log_transformed is ignored even though it is set to True.
    :type is_log_transformed: bool, optional
    :param int max_ft_freq: The maximum frequency order for the Fourier transformation.
    :type max_ft_freq: int, optional
    """
    def __init__(self, include_holidays_exog=True,
                 p=2,
                 q=2,
                 is_log_transformed=True,
                 max_ft_freq=3):

        super(LADStructuralHyperParams, self).__init__(
            model_name="LADStructuralModel",
            include_holidays_exog=include_holidays_exog,
            p=p,
            q=q,
            is_log_transformed=is_log_transformed,
            max_ft_freq=max_ft_freq,
        )


class LADStructuralError(Exception):
    """
    Exception class for Luminaire structural anomaly detection model.

    """
    def __init__(self, message):
        message = f'LAD structural failed! Error: {message}'

        super(LADStructuralError, self).__init__(message)


class LADStructuralModel(BaseModel):
    """
    A LAD structural time series model.

    :param dict hyper_params: Hyper parameters for Luminaire structural modeling.
        See :class:`luminaire.optimization.hyperparameter_optimization.HyperparameterOptimization` for detailed
        information.
    :param str freq: The frequency of the time-series. A `Pandas offset`_ such as 'D', 'H', or 'M'.
    :param min_ts_length: The minimum required length of the time series for training.
    :type min_ts_length: int, optional
    :param max_ts_length: The maximum required length of the time series for training.
    :type max_ts_length: int, optional
    :param min_ts_mean: Minimum average values in the most recent window of the time series. This optional parameter
        can be used to avoid over-alerting from noisy low volume time series.
    :type min_ts_mean: float, optional
    :param min_ts_mean_window: Size of the most recent window to calculate min_ts_mean.
    :type min_ts_mean_window: int, optional

    .. Note :: This class should be used to manually configure the structural model. Exact configuration parameters
        can be found in `luminaire.hyperparameter_optimization.HyperparameterOptimization`. Optimal configuration
        can be obtained by using LAD hyperparameter optimization.

    .. _statsmodels docs: http://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.html
    .. _Pandas offset: https://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases

    >>> hyper = {"include_holidays_exog": 0, "is_log_transformed": 1, "max_ft_freq": 2, "p": 5, "q": 1}
    lad_struct_model = LADStructuralModel(hyper_params=hyper, freq='D')
    >>> lad_struct_model
    <luminaire.model.lad_structural.LADStructuralModel object at 0x103efe320>
    """

    __version__ = "2.0"

    _target_metric = 'raw'
    _imputed_metric = 'interpolated'
    _sig_level = 0.10
    _sig_level_extreme = 0.001

    def __init__(self,
                 hyper_params: LADStructuralHyperParams().params or None,
                 freq,
                 min_ts_length=None,
                 max_ts_length=None,
                 min_ts_mean=None,
                 min_ts_mean_window=None,
                 **kwargs):

        self.hyper_params = hyper_params

        max_scoring_length_dict = {
            'H': 48,
            'D': 10,
            'W': 8, 'W-SUN': 8, 'W-MON': 8, 'W-TUE': 8, 'W-WED': 8, 'W-THU': 8, 'W-FRI': 8, 'W-SAT': 8,
            'M': 24, 'MS': 24,
        }
        self.max_scoring_length = max_scoring_length_dict.get(freq)

        fit_diagnostic_lag_dict = {
            'H': 144 * 2,
            'D': 7 * 4,
            'W': 12, 'W-SUN': 12, 'W-MON': 12, 'W-TUE': 12, 'W-WED': 12, 'W-THU': 12, 'W-FRI': 12, 'W-SAT': 12,
            'M': 24, 'MS': 24,
        }
        self._fit_diagnostic_lag = fit_diagnostic_lag_dict.get(freq)

        super(LADStructuralModel, self).__init__(freq=freq, min_ts_mean=min_ts_mean,
                                                 min_ts_mean_window=min_ts_mean_window,
                                                 min_ts_length=min_ts_length, max_ts_length=max_ts_length,
                                                 **hyper_params, **kwargs)

    @classmethod
    def _signals(cls, idx, m, n):
        """
        This function computes the sinusoids given the significant frequencies
        :param list idx: A list containing the significat frequency indices obtained from the spectral density plot in fourier_extp()
        :param int m: Specifying the current frequency
        :param int n: Specifying the length of the time series
        :return: A numpy array containing the sinusoids corresponding to the significant frequencies
        """
        import numpy as np
        signal = []

        # Generating all the frequencies from a time series of length n
        fs = np.fft.fftfreq(n)

        # Loop through the frequencies in idx
        for i in idx:
            freq = fs[i]

            # Computing the sinusoids for the ith frequency
            signal.append(np.cos(2 * np.pi * m * freq) + complex(0, np.sin(2 * np.pi * m * freq)))
        return np.array(signal)

    @classmethod
    def _inv_fft(cls, n_extp, n, idx, a):
        """
        This function computes the inverse Fourier transform given the significant frequencies obtained from the spectral
        density plot in fourier_extp()
        :param int n_extp: Specifying the length of the time series + the length of the forecast period
        :param int n: Int specifying the length of the time series
        :param list idx: A list containing the significant frequency indices obtained from the spectral density plot in fourier_extp()
        :param list a: A list containing the coefficient of the significant frequencies in the Fourier transform
        :return: A numpy array containing the inverse transformation from the fourier transformation of the original
        time series
        """
        import numpy as np
        ts = []
        for i in range(0, n_extp):
            # Sinusoid for the ith frequency
            s_array = cls._signals(idx, i, n)

            # Computing the inverse Fouries transformation term for the significant coefficients obtained from the
            # spectral density
            ts.append(np.sum(a * s_array) // n)
        return np.array(ts)

    @classmethod
    def _fourier_extp(cls, series=None, max_trun=None, forecast_period=None):
        """
        This function approximates the input time series using a Fourier and an inverse Fourier transformation using
        incremental
        frequencies (appending next two frequencies every time)
        :param list series: A list containing the input time series
        :param int max_trun: Specifying the maximum allowed frequency to consider for the Fourier approximation
        :param int forecast_period: Specifying the number of points to extrapolate using the fourier transform
        :return: A list of list where every column contains the Fourier approximation of the input time series using (2 * column number)
        many frequencies
        """
        import numpy as np
        import copy
        n = len(series)

        smoothing_loc = np.where((series < np.mean(series) - 3 * np.std(series)) | (series > np.mean(series)
                                                                                    + 3 * np.std(series)))
        smoothed_series = copy.deepcopy(series)
        if len(smoothing_loc[0]) > 0:
            for idx in smoothing_loc[0]:
                smoothed_series[idx] = np.mean(smoothed_series[max(0, idx - 6): max(0, idx - 1)]) if idx > 5 \
                    else smoothed_series[idx]

        iyf = []

        # Generating the indices based on odd and event number of terms in the time series
        if int(n) % 2 != 0:
            all_idx = np.arange(1, n // 2 + 1)
        else:
            all_idx = np.arange(1, n // 2)

        # Performing Fourier transformation
        yf = np.fft.rfft(smoothed_series)

        # Spectral density for the fourier transformation (to identify the significant frequencies)
        psd = abs(yf[all_idx]) ** 2 + abs(yf[-all_idx]) ** 2
        psd_sorted = np.copy(psd)
        psd_sorted[::-1].sort()

        max_trun = min(max_trun, max(len(psd_sorted) - 1, 0))

        # Computing inverse Fourier transformation by appending next two significant frequencies up to (2 * max_trun)
        # frequencies

        idx = all_idx[np.where(psd > psd_sorted[max_trun])[0]]
        idx = np.concatenate((np.array([0]), idx), axis=0)
        a = yf[idx]

        # Storing the inverse Fourier transformations with (2 * trun) many frequencies
        iyf.append(cls._inv_fft(n + forecast_period, n, idx, a))

        return np.array(iyf)

    def _seasonal_arima(self, endog=None, exog=None, p=None, d=None, q=None, imodels=None, include_holidays=None,
                        ift_matrix=None, stepwise_fit=None, optimize=None):
        """
        This function runs the ARIMA model with different Fourier transformations based on different number of
        frequencies.
        :param pandas.Dttaframe endog: A pandas dataframe storing the endogenous time series
        :param pandas.Dttaframe exog: A pandas dataframe storing the exogenous pulses obtained through the Fourier transformation
        and / or the a binary one hot encoding different US holidays
        :param int p: A tuple containing the minimum and maximum of the auto-regressive terms to be considered in the model
        :param int d: A tuple containing the minimum and maximum of the differencing to be considered in the model
        :param int q: A tuple containing the minimum and maximum of the moving average terms to be considered in the model
        :param int imodels: The current model run based on the current exogenous obtained through first imodels*2 many most
        relevant frequencies from the Fourier transform
        :param bool include_holidays: Whether to consider holidays as exogenous
        :param list ift_matrix: A list of list All exogenous variables where the ith column is the inverse Fourier
        transformation of the time series with first i*2 most relevant frequencies
        :param int train_len: Storing the length of the time series
        :param int pred_len: Storing the length of the future time points to predict
        :param bool arima_error: Storing whether there is any exception occurred in the auto_arima run
        :param list stepwise_fit: A list storing different model object
        :param bool optimize: Flag to identify whether called from hyperparameter optimization
        :param list x_pred: list storing exogenous variable corresponding to the time point to predict
        :return:
        """

        import numpy as np
        import statsmodels.tsa.arima_model as arima

        # Extract the exogenous variable generated based on (imodels * 2) number of most significant
        # frequencies
        if imodels > 0:
            fourier_exog = ift_matrix[0].reshape(-1, 1)[:, 0].reshape(-1, 1)
            if not include_holidays:
                exog = np.real(fourier_exog)
            else:
                exog['fourier_feature'] = np.float64(np.real(fourier_exog[:, 0]))

        # This check is required due to a bug in statsmodel arima which inflates the predictions and std error
        # for time series containing only 0's. Can be removed if fixed in a later version of statsmodel
        # or pyramid
        if np.count_nonzero(endog) == 0:
            idx_max = len(endog) // 2
            idx = int(np.random.randint(0, idx_max, 1)[0])
            endog[idx] = abs(np.random.normal(0, 1e-3, 1)[0])

        try:
            stepwise_fit.append(arima.ARIMA(endog=endog, exog=exog,
                                            order=(p, d, q)).fit(seasonal=False, trace=False,
                                                                 method='css',
                                                                 solver='bfgs',
                                                                 error_action='ignore',
                                                                 stepwise_fit=True,
                                                                 warn_convergence=False,
                                                                 disp=False))
        except Exception as e:
            raise LADStructuralError(message=str(e))

        return 0

    @classmethod
    def _get_exog_data(cls, exog_start, exog_end, index):
        """
        This function gets the exogenous data for the specified index.
        :param pandas.Timestamp exog_start: Start date for the exogenous data
        :param pandas.Timestampexog_end: End date for the exogenous data
        :param list[pandas.Timestamp] index: List of indices
        :return: Exogenous data for the given list of index
        :rtype: pandas.DataFrame
        """
        holiday_calendar = LADHolidays()
        holiday_series = holiday_calendar.holidays(start=exog_start, end=exog_end, return_name=True)
        return (pd.DataFrame({'Holiday': holiday_series, 'Ones': 1})
                .pivot(columns='Holiday', values='Ones')
                .reindex(index)
                .fillna(0)
                )

    def _fit(self, endog, endog_end, min_ts_mean, min_ts_mean_window, include_holidays=False,
             min_ts_length=None, max_ft_freq=None, exog_data=None, optimize=None):
        """
        This function implements the fourier transformation to model the periodicities and implements ARIMA model with
        different order of differencing, MA and AR terms to generate the optimal prediction and anomaly detection.
        :param list endog: A list containing the time series input
        :param str endog_end: pandas datetime containing the last timestamp of the input time series
        :param float raw_actual: Containing the actual value of the execution date
        :param float raw_actual_previous: Containing the actual value of the day before execution date
        :param float interpolated_actual: Containing the interpolated value of the execution date
        :param pandas.Dttaframe pred_instance_date: pandas datetime containing the prediction timestamp
        :param float min_ts_mean: The minimum mean value of the time series required for the model to run. For data that
        originated as integers (such as counts), the ARIMA model can behave erratically when the numbers are small. When
        this parameter is set, any time series whose mean value is less than this will automatically result in a model
        failure, rather than a mostly bogus anomaly.
        :param int min_ts_mean_window: The number of observations (anchored to the end of the time series) to use when
        applying the min_ts_mean rule. Default is None, which performs the calculation on the entire time series.
        :param bool include_holidays: Whether to include holidays as exogenous variables in the regression. Holidays
        are defined in :class:`~luminaire.model.model_utils.LADsHolidays`
        :param int min_ts_length: Specifying the minimum required length of the time series for training
        :param int max_ft_freq: The maximum number of frequencies under consideration for the Fourier transformation.
        :param bool optimize: Flag to identify whether called from hyperparameter optimization
        :return: A dictionary containing the anomaly flag, details of the prediction data (timestamp, raw, interpolated)
        lower and upper bound of the confidence interval, flag whether holidays are included in the model as exogenous
        """
        import numpy as np
        from pykalman import KalmanFilter
        import warnings
        warnings.filterwarnings('ignore')

        p, q = self._params['p'], self._params['q']
        freq = self._params['freq']
        pred_len = self.max_scoring_length
        x_matrix_train = None
        x_matrix_score = None

        # set exogenous (holiday) variables for input data
        if include_holidays and len(endog) + pred_len > 385:
            exog = exog_data.loc[endog.index.min():endog_end]
        else:
            include_holidays = False
            exog = None

        if min_ts_length is not None and len(endog) < min_ts_length:
            raise ValueError('TimeSeries length less than minimum length specified')

        if min_ts_mean is not None:
            if (min_ts_mean_window is not None and endog[-min_ts_mean_window:].fillna(0).mean() < min_ts_mean) or \
                    (min_ts_mean_window is None and endog.fillna(0).mean() < min_ts_mean):
                raise ValueError('Metric values too small to model.')

        # Smoothing the given time series as a pre-processing for modeling seasonalities through Fourier
        # transformation
        kf = KalmanFilter()
        endog_smoothed, filtered_state_covariances = kf.em(endog).smooth(endog)
        endog_smoothed = endog_smoothed[:, 0]

        endog, diff_order, actual_previous_per_diff = DataExploration._stationarizer(endog=pd.Series(endog),
                                                                                    diff_min=0,
                                                                                    diff_max=1,
                                                                                    obs_incl=False)
        if diff_order:
            endog_smoothed = np.diff(endog_smoothed)

        if freq == 'D':
            complete_cycle = int(len(endog) / 7)
            endog = endog[- (complete_cycle * 7):]
            endog_smoothed = endog_smoothed[- (complete_cycle * 7):]
        elif freq == 'H':
            complete_cycle = int(len(endog) / 24)
            endog = endog[- (complete_cycle * 24):]
            endog_smoothed = endog_smoothed[- (complete_cycle * 24):]

        exog = exog.iloc[-len(endog):] if exog is not None else None

        if include_holidays:
            exog = exog.loc[:, (exog != 0).any(axis=0)]
            ext_training_features = list(exog.columns)
        else:
            ext_training_features = None

        stepwise_fit = []

        # Updating the user specified maximum number of frequencies to consider for the Fourier transformation
        # based on the length of the smoothed endogenous variable
        max_ft_freq = int(min(max_ft_freq, len(endog_smoothed) / 4))

        # Running the Fourier transformation extrapolating one point ahead in future that is going to be used
        # for predicting

        if max_ft_freq > 0:
            x_matrix = self._fourier_extp(series=endog_smoothed, max_trun=(2 * max_ft_freq),
                                          forecast_period=pred_len)
            if not optimize and np.all(x_matrix[0] == x_matrix[0][0]):
                x_matrix_train = None
                x_matrix_score = None
                max_ft_freq = 0
            else:
                x_matrix_train = x_matrix[:, :(x_matrix.shape[1] - pred_len)]
                x_matrix_score = x_matrix[:, (x_matrix.shape[1] - pred_len):]


        self._seasonal_arima(endog=endog, exog=exog, p=p, d=0, q=q, imodels=max_ft_freq,
                             include_holidays=include_holidays, ift_matrix=x_matrix_train,
                             stepwise_fit=stepwise_fit, optimize=optimize)
        model = stepwise_fit[0]

        seasonal_feature_scoring = x_matrix_score[0, :].tolist() if not x_matrix_score is None else None

        result = {
            'model': model,
            'diff_order': diff_order,
            'seasonal_feature_scoring': seasonal_feature_scoring,
            'ext_training_features': ext_training_features,
        }

        p_selected = model.k_ar if hasattr(model, 'k_ar') else 0
        d_selected = diff_order
        q_selected = model.k_ma if hasattr(model, 'k_ma') else 0
        order = (p_selected, d_selected, q_selected)

        return result, order

    def _training(self, data, ts_start, ts_end, min_ts_length=None, min_ts_mean=None, min_ts_mean_window=None,
                  max_ft_freq=None, include_holidays=None, optimize=None, **kwargs):
        """
        This function performs the training for the input time series

        :param pandas.DataFrame data: Input time series data
        :param str ts_start: Start of the time series data
        :param str ts_end: End of the time series data
        :param int min_ts_length: Minimum required length for the time series data
        :param float min_ts_mean: Minimum mean value for the time series data for training over a fixed length window
        (min_ts_mean_window)
        :param int min_ts_mean_window: Length of the window to check min_ts_mean
        :param int max_ft_freq: Maximum number of frequency for the Fouries transformation
        :param bool is_log_transformed: Flag for log transformation
        :param bool optimize: Flag to identify whether called from hyperparameter optimization
        :return: Trained model and optimal LAD structural model order (p, d, q)
        :rtype: tuple[dict, tuple[int]]
        """
        from numpy.linalg import LinAlgError

        freq = self._params['freq']

        try:

            if data is None:
                raise ValueError('Not enough data to train due to recent change point')

            endog = data[self._imputed_metric]

            index = pd.date_range(start=ts_start, end=ts_end, freq=freq)  # Holidays are always daily.

            exog_data = self._get_exog_data(ts_start, ts_end, index) if self._params['include_holidays_exog'] else None

            # always run the model first without holiday exogenous variables
            result, order = self._fit(endog=endog, endog_end=ts_end, min_ts_mean=min_ts_mean,
                                      min_ts_mean_window=min_ts_mean_window, include_holidays=include_holidays,
                                      min_ts_length=min_ts_length, max_ft_freq=max_ft_freq, exog_data=exog_data,
                                      optimize=optimize)

            result['training_tail'] = data.loc[:ts_end].values.tolist()[-3:]

        except(LinAlgError, ValueError, LADStructuralError) as e:
            result = {'ErrorMessage': str(e)}
            return result, None

        return result, order

    def train(self, data, optimize=False, **kwargs):
        """
        This function trains a structural LAD model for a given time series.

        :param pandas.DataFrame data: Input time series data
        :param bool optimize: Flag to identify whether called from hyperparameter optimization
        :type optimize: bool, optional
        :return: success flag, the model date and the trained lad structural model object
        :rtype: tuple[bool, str, LADStructuralModel object]

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
        >>> hyper = {"include_holidays_exog": 0, "is_log_transformed": 0, "max_ft_freq": 2, "p": 5, "q": 1}
        >>> de_obj = DataExploration(freq='D', is_log_transformed=0)
        >>> data, pre_prc = de_obj.profile(data)
        >>> pre_prc
        {'success': True, 'trend_change_list': ['2020-04-01 00:00:00'], 'change_point_list': ['2020-03-16 00:00:00'],
        'is_log_transformed': 0, 'min_ts_mean': None, 'ts_start': '2020-01-01 00:00:00',
        'ts_end': '2020-06-07 00:00:00'}
        >>> lad_struct_obj = LADStructuralModel(hyper_params=hyper, freq='D')
        >>> model = lad_struct_obj.train(data=data, **pre_prc)

        >>> model
        (True, '2020-06-07 00:00:00', <luminaire.model.lad_structural.LADStructuralModel object at 0x126edf588>)
        """

        result, order = self._training(data=data,
                                       min_ts_length=self._params['min_ts_length'],
                                       min_ts_mean_window=self._params['min_ts_mean_window'],
                                       max_ft_freq=self._params['max_ft_freq'],
                                       include_holidays=self._params['include_holidays_exog'],
                                       optimize=optimize,
                                       **kwargs
                                       )

        self.hyper_params['is_log_transformed'] = kwargs['is_log_transformed']
        result['training_end_date'] = kwargs['ts_end']
        result['freq'] = self._params['freq']

        success = False if 'ErrorMessage' in result else True

        return success, kwargs['ts_end'], LADStructuralModel(hyper_params=self.hyper_params, **result)

    @classmethod
    def _predict(cls, model, is_log_transformed,
                 raw_actual, interpolated_actual,
                 training_end=None, seasonal_feature_scoring=None, pred_date=None, order_of_diff=None,
                 training_tail=None, ext_training_features=None, pred_len=None, freq=None,
                 include_holidays_exog=None):
        """
        This function performs the prediction and anomaly detection for a given prediction date and a time point

        :param python object model: LAD structural model object
        :param bool is_log_transformed: Flag for log transformation
        :param float raw_actual: Observed value of the time point
        :param float interpolated_actual: interpolated value of the time point
        :param str training_end: Last time series timestamp
        :param list seasonal_feature_scoring: Fourier features
        :param str pred_date: Prediction date
        :param int order_of_diff: Order of differencing for the nonstationarity property of the given time series
        :param list training_tail: Padding from latest time series observed values for prediction
        :param pandas.DataFrame ext_training_features: External exogenous variables
        :param int pred_len: Length of time the prediction need to be generated for
        :param str freq: Frequency of the observed time series
        :param bool include_holidays_exog: Flag to include holidays as exogenous in the model
        :return: Model result
        :rtype: dict
        """

        import numpy as np
        import pandas as pd
        import scipy.stats as st
        from numpy.linalg import LinAlgError
        import math

        alpha = cls._sig_level
        alpha_extreme = cls._sig_level_extreme

        include_holidays_exog = include_holidays_exog if ext_training_features else 0

        index = pd.date_range(start=training_end, end=pred_date, freq=freq)[1:]  # Holidays are always daily.

        pred_exog = cls._get_exog_data(pred_date, pred_date, index) if include_holidays_exog else None

        if pred_exog is not None and set(pred_exog.columns.values) != set(ext_training_features):
            missing_col_list = list(set(ext_training_features) - set(pred_exog.columns.values))
            common_cols = list(set(ext_training_features).intersection(set(pred_exog.columns.values)))
            temp_df = pred_exog[common_cols]
            missing_feat_df = pd.DataFrame(np.zeros([len(pred_exog), len(missing_col_list)]),
                                           columns=missing_col_list, index=pred_exog.index.values)
            pred_exog = pd.concat([temp_df, missing_feat_df], axis=1)
            pred_exog = pred_exog[ext_training_features]

        freq = "1" + freq if not any(char.isdigit() for char in freq) else freq

        forecast_ndays = int((pred_date - pd.Timestamp(training_end)) / pd.Timedelta(freq))
        model_freshness = forecast_ndays / float(pred_len)

        try:
            if forecast_ndays > pred_len:
                raise ValueError('Current trained model object expired')

            float_min = 1e-10

            # set exogenous (holiday) variables for input data
            if include_holidays_exog:
                pred_exog = pred_exog.loc[pd.Timestamp(training_end) + pd.Timedelta(freq): pred_date]
            else:
                pred_exog = None

            if seasonal_feature_scoring:
                if not include_holidays_exog:
                    pred_exog = seasonal_feature_scoring[:forecast_ndays]
                else:
                    pred_exog['fourier_feature'] = seasonal_feature_scoring[:forecast_ndays]

            forecast = list(model.forecast(steps=forecast_ndays, alpha=alpha, exog=pred_exog))
            interpolated_training_data = list(zip(*training_tail))[1]

            for order in list(reversed(range(order_of_diff))):
                training_data_diff = np.diff(interpolated_training_data,
                                             order) if order > 0 else interpolated_training_data

                forecast_diff_mean = [training_data_diff[-1]]
                forecast_diff_ci = []

                for i in range(forecast_ndays):
                    forecast_diff_mean.append(forecast_diff_mean[-1] + forecast[0][i])
                    forecast_diff_ci.append([forecast_diff_mean[-1] -
                                             (st.norm.ppf(1 - (alpha / 2.0)) * forecast[1][i]),
                                             forecast_diff_mean[-1] +
                                             (st.norm.ppf(1 - (alpha / 2.0)) * forecast[1][i])])
                forecast[0] = forecast_diff_mean[1:]
                forecast[2] = forecast_diff_ci

            if is_log_transformed:
                transformed_back_forecast = np.exp(forecast[0][-1] + ((forecast[1][-1] ** 2) / 2.0)) - 1
                transformed_back_std_err = np.sqrt((np.exp(forecast[1][-1] ** 2) - 1) * (np.exp((2 * forecast[0][-1]) +
                                                                                                (forecast[1][
                                                                                                     -1] ** 2))))
                transformed_back_CILower = transformed_back_forecast - \
                                           st.norm.ppf(1 - (alpha / 2.0), 0, transformed_back_std_err) \
                    if transformed_back_std_err != 0 else transformed_back_forecast
                transformed_back_CIUpper = transformed_back_forecast + \
                                           st.norm.ppf(1 - (alpha / 2.0), 0, transformed_back_std_err) \
                    if transformed_back_std_err != 0 else transformed_back_forecast
                transformed_back_interpolated_actual = float(np.exp(interpolated_actual) - 1)
            if np.sum(np.isnan(forecast[0][-1])) or np.isnan(forecast[1][-1]):
                raise ValueError('Predicted null value')

            if is_log_transformed:
                zscore = (transformed_back_interpolated_actual -
                          transformed_back_forecast) / max(float(transformed_back_std_err), float_min)

                anomaly_probability = (2 * st.norm(0, 1).cdf(abs(zscore))) - 1
                if math.isnan(anomaly_probability) or math.isnan(transformed_back_CILower) \
                        or math.isnan(transformed_back_CIUpper):
                    raise ValueError('Either Anomaly probability or CILower or CIUpper is NaN under log transform')
                down_anomaly_probability = 1 - st.norm(0, 1).cdf(zscore)
                up_anomaly_probability = st.norm(0, 1).cdf(zscore)

                result = {'Success': True,
                          'IsLogTransformed': is_log_transformed,
                          'LogTransformedAdjustedActual': interpolated_actual,
                          'LogTransformedPrediction': float(forecast[0][-1]),
                          'LogTransformedStdErr': float(forecast[1][-1]),
                          'LogTransformedCILower': float(forecast[2][-1][0]),
                          'LogTransformedCIUpper': float(forecast[2][-1][1]),
                          'AdjustedActual': transformed_back_interpolated_actual,
                          'Prediction': float(transformed_back_forecast) if not float(
                              transformed_back_forecast) == float('inf') else 0.0,
                          'StdErr': float(transformed_back_std_err) if not float(
                              transformed_back_std_err) == float('inf') else 0.0,
                          'CILower': float(transformed_back_CILower) if not float(
                              transformed_back_CILower) == float('-inf') else 0.0,
                          'CIUpper': float(transformed_back_CIUpper) if not float(
                              transformed_back_CIUpper) == float('inf') else 0.0,
                          'ConfLevel': float(1.0 - alpha) * 100,
                          'ExogenousHolidays': include_holidays_exog,
                          'IsAnomaly': bool(anomaly_probability > 1 - alpha),
                          'IsAnomalyExtreme': bool(anomaly_probability > 1 - alpha_extreme),
                          'AnomalyProbability': 1 if raw_actual is None else float(anomaly_probability),
                          'DownAnomalyProbability': 1 if raw_actual is None else float(down_anomaly_probability),
                          'UpAnomalyProbability': 1 if raw_actual is None else float(up_anomaly_probability),
                          'ModelFreshness': model_freshness}

            else:
                zscore = (interpolated_actual - forecast[0][-1]) / max(float(forecast[1][-1]), float_min)

                anomaly_probability = (2 * st.norm(0, 1).cdf(abs(zscore))) - 1
                if math.isnan(anomaly_probability) or math.isnan(forecast[2][-1][0]) or math.isnan(forecast[2][-1][1]):
                    raise ValueError('Either Anomaly probability or CILower or CIUpper is NaN')

                down_anomaly_probability = 1 - st.norm(0, 1).cdf(zscore)
                up_anomaly_probability = st.norm(0, 1).cdf(zscore)

                result = {'Success': True,
                          'IsLogTransformed': is_log_transformed,
                          'AdjustedActual': interpolated_actual,
                          'Prediction': float(forecast[0][-1]) if not float(
                              forecast[0][-1]) == float('inf') else 0.0,
                          'StdErr': float(forecast[1][-1]) if not float(
                              forecast[1][-1]) == float('inf') else 0.0,
                          'CILower': float(forecast[2][-1][0]) if not float(
                              forecast[2][-1][0]) == float('-inf') else 0.0,
                          'CIUpper': float(forecast[2][-1][1]) if not float(
                              forecast[2][-1][1]) == float('inf') else 0.0,
                          'ConfLevel': float(1.0 - alpha) * 100,
                          'ExogenousHolidays': include_holidays_exog,
                          'IsAnomaly': bool(anomaly_probability > 1 - alpha),
                          'IsAnomalyExtreme': bool(anomaly_probability > 1 - alpha_extreme),
                          'AnomalyProbability': 1 if raw_actual is None else float(anomaly_probability),
                          'DownAnomalyProbability': 1 if raw_actual is None else float(down_anomaly_probability),
                          'UpAnomalyProbability': 1 if raw_actual is None else float(up_anomaly_probability),
                          'ModelFreshness': model_freshness}

        except (LinAlgError, ValueError, LADStructuralError) as e:
            result = {'Success': False,
                      'AdjustedActual': interpolated_actual,
                      'ErrorMessage': str(e)}

        return result

    @classmethod
    def _scoring(cls, model, observed_value, pred_date, training_end=None,
                 seasonal_feature_scoring=None,
                 is_log_transformed=None, order_of_diff=None,
                 training_tail=None, ext_training_features=None, pred_len=None, freq=None,
                 include_holidays_exog=None):
        """
        This function performs scoring using the LAD structural model object

        :param python object model: LAD structural model object
        :param float observed_value: Observed time series value
        :param str pred_date: Prediction date
        :param str training_end: Last time series timestamp
        :param list seasonal_feature_scoring: Fourier features
        :param bool is_log_transformed: Flag for log transformation
        :param int order_of_diff: Order of differencing for the nonstationarity property of the given time series
        :param list training_tail: Padding from latest time series observed values for prediction
        :param pandas.DataFrame ext_training_features: External exogenous variables
        :param int pred_len: Length of time the prediction need to be generated for
        :param str freq: Frequency of the observed time series
        :param bool include_holidays_exog: Flag to include holidays as exogenous in the model
        :return: Model result
        :rtype: dict
        """

        import pandas as pd
        import numpy as np

        # Date to predict
        pred_date = pd.Timestamp(pred_date)

        if is_log_transformed:
            interpolated_actual = 0 if (observed_value is None or observed_value <= 0) else np.log(observed_value+1)
        else:
            interpolated_actual = 0 if observed_value is None else observed_value

        result = cls._predict(model=model, is_log_transformed=is_log_transformed,
                              raw_actual=observed_value, interpolated_actual=interpolated_actual,
                              training_end=training_end, seasonal_feature_scoring=seasonal_feature_scoring,
                              pred_date=pred_date, order_of_diff=order_of_diff, training_tail=training_tail,
                              ext_training_features=ext_training_features, pred_len=pred_len,
                              freq=freq, include_holidays_exog=include_holidays_exog)

        return result

    def score(self, observed_value, pred_date, **kwargs):
        """
        This function scores a value observed at a data date given a trained LAD structural model object.

        :param float observed_value: Observed time series value on the prediction date.
        :param str pred_date: Prediction date. Needs to be in yyyy-mm-dd or yyyy-mm-dd hh:mm:ss format.
        :return: Anomaly flag, anomaly probability, prediction and other related metrics.
        :rtype: dict

        >>> model
        <luminaire.model.lad_structural.LADStructuralModel object at 0x11c1c3550>
        >>> model._params['training_end_date'] # Last data date for training time series
        '2020-06-07 00:00:00'

        >>> model.score(2000 ,'2020-06-08')
        {'Success': True, 'IsLogTransformed': 0, 'AdjustedActual': 2000, 'Prediction': 1943.20426163425,
        'StdErr': 93.084646777553, 'CILower': 1785.519523590432, 'CIUpper': 2100.88899967807, 'ConfLevel': 90.0,
        'ExogenousHolidays': 0, 'IsAnomaly': False, 'IsAnomalyExtreme': False, 'AnomalyProbability': 0.42671448831719605,
        'DownAnomalyProbability': 0.286642755841402, 'UpAnomalyProbability': 0.713357244158598, 'ModelFreshness': 0.1}
        >>> model.score(2500 ,'2020-06-09')
        {'Success': True, 'IsLogTransformed': 0, 'AdjustedActual': 2500, 'Prediction': 2028.989933854948,
        'StdErr': 93.6623172459385, 'CILower': 1861.009403637476, 'CIUpper': 2186.97046407242, 'ConfLevel': 90.0,
        'ExogenousHolidays': 0, 'IsAnomaly': True, 'IsAnomalyExtreme': True, 'AnomalyProbability': 0.9999987021695071,
        'DownAnomalyProbability': 6.489152464261849e-07, 'UpAnomalyProbability': 0.9999993510847536,
        'ModelFreshness': 0.2}

        """

        result = self._scoring(model=self._params['model'], observed_value=observed_value, pred_date=pred_date,
                              training_end=self._params['training_end_date'],
                              seasonal_feature_scoring=self._params['seasonal_feature_scoring'],
                              is_log_transformed=self._params['is_log_transformed'],
                              order_of_diff=self._params['diff_order'],
                              training_tail=self._params['training_tail'],
                              ext_training_features=self._params['ext_training_features'],
                              pred_len=self.max_scoring_length, freq=self._params['freq'],
                              include_holidays_exog=self._params['include_holidays_exog'])

        return result
