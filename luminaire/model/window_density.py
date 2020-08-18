from luminaire.model.base_model import BaseModel, BaseModelHyperParams
from luminaire.exploration.data_exploration import DataExploration


class WindowDensityHyperParams(BaseModelHyperParams):
    """
    Hyperparameter class for Luminaire Window density model.

    :param str freq: The frequency of the time-series. Luminaire supports default configuration for 'S', 'M', 'QM',
        'H', 'D'. Any other frequency type should be specified as 'custom' and configuration should be set manually.
    :param int ignore_window: ignore a time window to be considered for training.
    :type ignore_window: int, optional
    :param float max_missing_train_prop: Maximum proportion of missing observation allowed in the training data.
    :type max_missing_train_prop: float, optional
    :param bool is_log_transformed: A flag to specify whether to take a log transform of the input data.
        If the data contain negatives, is_log_transformed is ignored even though it is set to True.
    :type is_log_transformed: bool, optional
    :param str baseline_type: A string flag to specify whether to take set a baseline as the previous sub-window from
        the training data for scoring or to aggregate the overall window as a baseline.
        Possible values:
            - "last_window"
            - "aggregated"
    :type baseline_type: str, optional
    :param str detection_method: A string that select between two window testing method.
        Possible values:
            - "kldiv" (KL-divergence)
            - "sign_test" (Wilcoxon sign rank test)

    :type detection_method: str, optional
    :param int min_window_length: Minimum size of the scoring window / a stable training sub-window length.
    :type min_window_length: int, optional
    .. Note :: This is not the minimum size of the whole training window which is the combination of stable sub-windows.
    :param int max_window_length: Maximum size of the scoring window / a stable training sub-window length.
    :type max_window_length: int, optional
    .. Note :: This is not the maximum size of the whole training window which is the combination of stable sub-windows.
    :param int window_length: Size of the scoring window / a stable training sub-window length.
    :type window_length: int, optional
    .. Note :: This is not the size of the whole training window which is the combination of stable sub-windows.
    :param int ma_window_length: Size of the window for detrending scoring window / stable training sub-windows through
        moving average method.
    :type ma_window_length: int, optional
    .. Note :: ma_window_length should be small enough to maintain the stable structure of the training / scoring window
        and large enough to remove the trend. The ideal size can be somewhere between (0.1 * window_length) and
        (0.25 * window length).
    :param str detrend_method: A string that select between two stationarizing method. Possible values:
            - "ma" (moving average based)
            - "diff" (differencing based).
    :type detrend_method: str, optional
    """
    def __init__(self,
                 freq='M',
                 ignore_window=None,
                 max_missing_train_prop=0.1,
                 is_log_transformed=False,
                 baseline_type="aggregated",
                 detection_method=None,
                 min_window_length=None,
                 max_window_length=None,
                 window_length=None,
                 ma_window_length=None,
                 detrend_method='ma'
                 ):
        # Detection method is KL divergence for high frequency data and sign test for low frequency data
        if not detection_method:
            detection_method = "kldiv" if freq in ['S', 'M', 'QM'] else "sign_test"

        # Pre-specification of the window lengths for different window frequencies with their min and max
        min_window_length_dict = {
            'S': 60 * 10,
            'M': 60 * 12,
            'QM': 4 * 24 * 7,
            'H': 12, 'D': 10,
        }
        max_window_length_dict = {
            'S': 60 * 60 * 24,
            'M': 60 * 24 * 84,
            'QM': 4 * 24 * 168,
            'H': 24 * 7, 'D': 90,
        }
        window_length_dict = {
            'S': 60 * 60,
            'M': 60 * 24,
            'QM': 4 * 24 * 14,
            'H': 24, 'D': 28,
        }
        ma_window_length_dict = {
            'S': 10 * 60,
            'M': 60,
            'QM': 4 * 4,
            'H': 12, 'D': 7,
        }

        if freq in ['S', 'M', 'QM', 'H', 'D']:
            min_window_length = min_window_length_dict.get(freq)
            max_window_length = max_window_length_dict.get(freq)
            window_length = window_length_dict.get(freq)
            ma_window_length = ma_window_length_dict.get(freq)

        super(WindowDensityHyperParams, self).__init__(
            model_name="WindowDensityModel",
            freq=freq,
            ignore_window=ignore_window,
            max_missing_train_prop=max_missing_train_prop,
            is_log_transformed=is_log_transformed,
            baseline_type=baseline_type,
            detection_method=detection_method,
            min_window_length=min_window_length,
            max_window_length=max_window_length,
            window_length=window_length,
            ma_window_length=ma_window_length,
            detrend_method=detrend_method
        )


class WindowDensityModel(BaseModel):
    """
    This model detects anomalous windows using KL divergence (for high frequency data) and Wilcoxon sign rank test
    (for low frequency data).

    :param dict hyper_params: Hyper parameters for Luminaire window density model.
        See :class:`luminaire.model.window_density.WindowDensityHyperParams` for detailed information.
    :return: Anomaly probability for the execution window and other related model outputs
    :rtype: list[dict]
    """

    __version__ = "0.1"

    def __init__(self,
                 hyper_params: WindowDensityHyperParams().params or None,
                 **kwargs):

        # Specifying the minimum and maximum number of training windows
        self.min_num_train_windows = 5
        self.max_num_train_windows = 10000
        self.hyper_params = hyper_params
        self.sig_level = 0.001

        super(WindowDensityModel, self).__init__(**hyper_params, **kwargs)

    def _volume_shift_detection(self, mean_list=None, sd_list=None, probability_threshold=0.5):
        """
        This function detects any significant shift in the training data volume using a Bayesian change point detection
        technique.

        :param list mean_list: The list of means from each training sub-window.
        :param list sd_list: The list of standard deviations from each training sub-window.
        :param float probability_threshold: Threshold for the probability value to be flagged as a change point.
        :return: Indices with significant vdata volume shift.
        :rtype: int
        """
        import numpy as np
        from bayesian_changepoint_detection import offline_changepoint_detection as offcd
        from functools import partial

        # Volume shift detection over the means of the training window
        q, p, pcp = offcd.offline_changepoint_detection(
            data=np.array(mean_list),
            prior_func=partial(offcd.const_prior, l=(len(mean_list) + 1)),
            observation_log_likelihood_function=offcd.gaussian_obs_log_likelihood,
            truncate=-10)

        mask_mean = np.append(0, np.exp(pcp).sum(0)) > probability_threshold

        # Volume shift detection over the standard deviations of the training window
        change_points = np.array(mask_mean).nonzero()
        last_mean_cp = change_points[0][-1] if len(change_points[0]) > 0 else []

        q, p, pcp = offcd.offline_changepoint_detection(
            data=np.array(sd_list),
            prior_func=partial(offcd.const_prior, l=(len(sd_list) + 1)),
            observation_log_likelihood_function=offcd.gaussian_obs_log_likelihood,
            truncate=-10)

        mask_sd = np.append(0, np.exp(pcp).sum(0)) > probability_threshold

        change_points = np.array(mask_sd).nonzero()
        last_sd_cp = change_points[0][-1] if len(change_points[0]) > 0 else []

        # Change point is the maximum obtained from mean list and the standard deviation list
        cdate = max(last_mean_cp, last_sd_cp)

        return cdate

    def _distance_function(self, data=None, called_for=None, baseline=None):
        """
        This function finds the distance of the given data from the baseline using KL divergence.

        :param list data: The list containing the scoring window (for scoring) / training sub-window (for training).
        :param str distance_method: The method to be used to calculate the distance between two datasets.
        :param str called_for: A flag to specify whether this function is called for training or scoring.
        :param list baseline: A list containing the base line to be compared with the given data.
        :return: KL divergence between two time windows.
        :rtype: float
        """
        import numpy as np
        import scipy.stats as stats
        float_min = 1e-50
        float_max = 1e50

        # If called for training, Kl divergence is performed over each pair of consecutive windows to create
        # the past anomaly scores
        if called_for == "training":
            distance = []
            for i in range(0, len(data) - 1):
                q = stats.kde.gaussian_kde(data[i])
                p = stats.kde.gaussian_kde(data[i + 1])

                ts_min = min(np.min(data[i]), np.min(data[i + 1]))
                ts_max = max(np.max(data[i]), np.max(data[i + 1]))

                density_domain = np.linspace(ts_min, ts_max, 1000)
                q = q(density_domain)
                p = p(density_domain)

                # approximating the zero probability regions to avoid divide by zero issue in KL divergence
                q[q == 0] = min(np.array(q)[np.array(q) > 0])
                p[p == 0] = min(np.array(p)[np.array(p) > 0])

                q = np.clip(q, float_min, float_max)
                p = np.clip(p, float_min, float_max)

                distance.append(stats.entropy(pk=p, qk=q))

        # If called for scoring, Kl divergence is performed between the scoring window and the baseline
        elif called_for == "scoring":
            q = stats.kde.gaussian_kde(baseline)
            p = stats.kde.gaussian_kde(data)

            ts_min = min(np.min(baseline), np.min(data))
            ts_max = max(np.max(baseline), np.max(data))

            density_domain = np.linspace(ts_min, ts_max, 1000)
            q = q(density_domain)
            p = p(density_domain)

            q[q == 0] = min(np.array(q)[np.array(q) > 0])
            p[p == 0] = min(np.array(p)[np.array(p) > 0])

            q = np.clip(q, float_min, float_max)
            p = np.clip(p, float_min, float_max)

            distance = stats.entropy(pk=p, qk=q)

        return distance

    def _training_data_truncation(self, sliced_training_data=None):
        """
        This function performs the truncation of the training data using the _volume_shift_detection function.

        :param list sliced_training_data: The list containing the training data.
        :return: Sliced training sample based on the most recent change point
        :rtype: list
        """
        import numpy as np

        # Change point detection is performed over the means and standard deviations of the sub windows
        window_means = []
        window_sds = []
        for ts in sliced_training_data:
            window_means.append(np.mean(ts))
            window_sds.append(np.std(ts))

        change_point = self._volume_shift_detection(mean_list=window_means, sd_list=window_sds)

        # Truncating the training data based on the last change point
        if change_point:
            sliced_training_data_truncated = sliced_training_data[change_point:]
            return sliced_training_data_truncated
        else:
            return sliced_training_data

    def _call_training(self, training_start, training_end, df=None, window_length=None, min_window_length=None,
                       max_window_length=None, min_num_train_windows=None, max_num_train_windows=None,
                       ignore_window=None, imputed_metric=None, detrend_method=None, **kwargs):
        """
        This function generates the baseline and training metrics to be used for scoring
        :param str training_start: Training start date.
        :param str training_end: Training end date.
        :param pandas.DataFrame df: Input training data frame.
        :param dict attributes: Model attributes.
        :param int window_length: The length of a training sub-window.
        :param int min_window_length: Minimum size of a stable training sub-window length.
        :param int max_window_length: Maximum size of a stable training sub-window length.
        :param int min_num_train_windows: Minimum number of training windows.
        :param int max_num_train_windows: Maximum number of training windows.
        :param ignore_window: ignore a time window to be considered for training.
        :param str imputed_metric: Column storing the time series values.
        :param str detrend_method: Detrend method "ma" or "diff" for nonstationarity.
        :return: Returns past anomaly scores based on training data, baseline and other related metrics.
        :rtype: tuple(list, float, float, int, list, float, dict, list)
        """
        import pandas as pd

        if training_start:
            # if a timeseries start date is provided, take the larger of this or the first timeseries value
            training_start = max(df.index.min(), pd.Timestamp(training_start))
        else:
            # take first value of timeseries by default
            training_start = str(df.index.min())

        training_window = [pd.to_datetime(training_start), pd.to_datetime(training_end)]

        if window_length < min_window_length:
            raise ValueError('Training window too small')
        if window_length > max_window_length:
            raise ValueError('Training window too large')
        n_windows = len(df[pd.to_datetime(training_start): pd.to_datetime(training_end)]) // window_length
        if n_windows < min_num_train_windows:
            raise ValueError('Too few training windows')
        if n_windows > max_num_train_windows:
            raise ValueError('Too many training windows')

        past_anomaly_scores, anomaly_scores_mean, anomaly_scores_sd, \
        detrend_order, baseline, ma_forecast_adj = self._anomalous_region_detection(input_df=df,
                                                                                    window_length=window_length,
                                                                                    training_window=training_window,
                                                                                    ignore_window=ignore_window,
                                                                                    value_column=imputed_metric,
                                                                                    called_for="training",
                                                                                    detrend_method=detrend_method)
        training_tail = df.loc[:training_window[1]].iloc[-window_length:]['interpolated'].to_list()

        return past_anomaly_scores, anomaly_scores_mean, anomaly_scores_sd, \
               detrend_order, baseline, ma_forecast_adj, training_tail, training_start, training_end


    def _get_model(self, input_df=None, training_window=None, window_length=None, ignore_window=None, value_column=None,
                  ma_window_length=None, detrend_method=None, baseline_type=None, detection_method=None):
        """
        This function runs the training process given the input parameters.
        :param pandas.DataFrame input_df: Input data containing the training and the scoring data.
        :param list training_window: A list containing the start and the end of the training window.
        :param int window_length: The length of a training sub-window / scoring window.
        :param int ignore_window: ignore a time window to be considered for training.
        :param str value_column: Column containing the values.
        :param int ma_window_length: Length of the moving average window to be used for detrending.
        :param str detrend_method: Selects between "ma" or "diff" detrend method.
        :param str baseline_type: Selects between "aggregated" or "last_window" baseline.
        :param str detection_method: Selects between "kldiv" or "sign_test" distance method.
        :return: Returns past anomaly scores based on training data, baseline and other related metrics.
        :rtype: tuple(list, float, float, int, list, float)
        """
        import numpy as np
        import pandas as pd
        from itertools import chain


        # Obtaining and slicing the training data based on the window size

        training_df = input_df[pd.to_datetime(training_window[0]): pd.to_datetime(training_window[1])]

        training_data = list(training_df[value_column])

        de_obj = DataExploration()
        sliced_training_data = de_obj._partition(training_data, window_length)
        sliced_training_data_normal = []

        # Cleaning the training data given a externally specified bad training sub-window
        for i in range(0, len(sliced_training_data)):
            if not (ignore_window is None):
                if (i + 1) not in ignore_window:
                    sliced_training_data_normal.append(sliced_training_data[i])
            else:
                sliced_training_data_normal.append(sliced_training_data[i])

        de_obj = DataExploration()

        # performing the stationarity test
        sliced_training_data_cleaned, detrend_order, ma_forecast_adj = de_obj._detrender(
            training_data_sliced=sliced_training_data_normal,
            ma_window_length=ma_window_length,
            significance_level=0.05,
            detrend_method=detrend_method, train_subwindow_len=window_length)

        # Obtain the past anomaly scores and the anomaly means and standard deviation if the detection method
        # is KL divergence
        if detection_method == "kldiv":
            past_anomaly_scores = np.array(self._distance_function(data=sliced_training_data_cleaned,
                                                                   called_for="training"))
            past_anomaly_scores = past_anomaly_scores[past_anomaly_scores <
                                                      np.percentile(past_anomaly_scores,
                                                                    90, interpolation='midpoint')].tolist()
            anomaly_scores_mean = np.mean(past_anomaly_scores)
            anomaly_scores_sd = np.std(past_anomaly_scores, ddof=1)
        else:
            past_anomaly_scores, anomaly_scores_mean, anomaly_scores_sd = None, None, None

        # If aggregated baseline type is specified, we take the whole training window as a baseline, else we
        # take the last training sub window from the sliced training data
        if baseline_type == "aggregated":
            sliced_training_data_cleaned = self._training_data_truncation(
                sliced_training_data=sliced_training_data_cleaned)
            if detection_method == "kldiv":
                baseline = list(chain.from_iterable(sliced_training_data_cleaned))
            elif detection_method == "sign_test":
                baseline = sliced_training_data_cleaned
        elif baseline_type == "last_window":
            baseline = sliced_training_data_cleaned[-1]

        return past_anomaly_scores, anomaly_scores_mean, anomaly_scores_sd, detrend_order, baseline, ma_forecast_adj

    def train(self, data, **kwargs):
        """
        Input time series for training.

        :param data: Input time series.
        :return: Training summary with a success flag.
        :rtype: tuple(bool, python model object)

        >>> data
                                raw interpolated
        index
        2017-10-02 00:00:00  118870       118870
        2017-10-02 01:00:00  121914       121914
        2017-10-02 02:00:00  116097       116097
        2017-10-02 03:00:00   94511        94511
        2017-10-02 04:00:00   68330        68330
        ...                     ...          ...
        2018-10-10 19:00:00  219908       219908
        2018-10-10 20:00:00  219149       219149
        2018-10-10 21:00:00  207232       207232
        2018-10-10 22:00:00  198741       198741
        2018-10-10 23:00:00  213751       213751
        >>> hyper_params = WindowDensityHyperParams(freq='H').params
        >>> wdm_obj = WindowDensityModel(hyper_params=hyper_params)
        >>> success, model = wdm_obj.train(data)

        >>> success, model
        (True, <luminaire.model.window_density.WindowDensityModel object at 0x7fd7c5a34e80>)
        """
        import numpy as np
        import pandas as pd

        freq = self._params['freq']
        min_num_train_windows = self.min_num_train_windows
        max_num_train_windows = self.max_num_train_windows
        ignore_window = self._params['ignore_window']
        if freq in ['S', 'M', 'QM', 'H', 'D']:
            min_window_length = self._params['min_window_length']
            max_window_length = self._params['max_window_length']
            window_length = self._params['window_length']
        else:
            min_window_length = self._params['min_window_length']
            max_window_length = self._params['max_window_length']
            window_length = self._params['window_length']
            if not min_window_length or not max_window_length or not window_length:
                raise ValueError(
                    'Training window length with min and max should be specified in case frequency not in the '
                    'specified list')
        is_log_transformed = self._params['is_log_transformed']
        max_missing_train_prop = self._params['max_missing_train_prop']
        detrend_method = self._params['detrend_method']
        target_metric = 'raw'
        imputed_metric = 'interpolated'
        if freq not in ['S', 'M', 'QM', 'H', 'D']:
            detection_method = self._params['detection_method']
            if not detection_method:
                raise ValueError('Detection method should be specified in case frequency not in the specified list')
            if detrend_method == 'ma':
                ma_window_length = self._params['ma_window_length']
                if not ma_window_length:
                    raise ValueError(
                        'Moving average window length should be specified for detrending for frequency not in the '
                        'specified list')

        if len(data) == 0:
            model = {'ErrorMessage': 'DataFrame length is 0'}
            success = False
            return success, WindowDensityModel(**model)

        if np.sum(np.isnan(data[target_metric])) > max_missing_train_prop:
            raise ValueError('Too few observed data in the training time series')
        else:
            de_obj = DataExploration()
            df = de_obj._kalman_smoothing_imputation(df=data, target_metric=target_metric, imputed_metric=imputed_metric)

        # Shift the interpolated value by +1 and get the log. This handles values with 0.
        if is_log_transformed:
            neg_flag = True if not data[data[target_metric] < 0].empty else False
            df[imputed_metric] = df[imputed_metric] if neg_flag else np.log(df[imputed_metric] + 1)

        past_anomaly_scores, anomaly_scores_mean, anomaly_scores_sd, detrend_order, baseline, ma_forecast_adj, \
        training_tail, training_start, training_end = self._call_training(df=df, window_length=window_length,
                                                                          min_window_length=min_window_length,
                                                                          max_window_length=max_window_length,
                                                                          min_num_train_windows=min_num_train_windows,
                                                                          max_num_train_windows=max_num_train_windows,
                                                                          ignore_window=ignore_window,
                                                                          imputed_metric=imputed_metric,
                                                                          detrend_method=detrend_method, **kwargs)

        success = True
        self.hyper_params['is_log_transformed'] = is_log_transformed
        model = {'ModelInstanceTimestamp': pd.Timestamp(training_end).time().strftime('%H:%M:%S'),
                 'TrainingStartDate': training_start,
                 'TrainingEndDate': training_end,
                 'PastAnomalyScores': past_anomaly_scores,
                 'AnomalyScoresMean': float(anomaly_scores_mean) if anomaly_scores_mean else None,
                 'AnomalyScoresSD': float(anomaly_scores_sd) if anomaly_scores_sd else None,
                 'NonStationarityOrder': detrend_order,
                 'Baseline': baseline,
                 'MovAvgForecastAdj': ma_forecast_adj,
                 'TrainingTail': training_tail
                 }

        return success, WindowDensityModel(hyper_params=self.hyper_params, **model)

    def _call_scoring(self, df=None, imputed_metric=None, anomaly_scores_mean=None, anomaly_scores_sd=None,
                      baseline=None, detrend_order=None, detrend_method=None, ma_forecast_adj=None, attributes=None,
                      training_tail=None):
        """
        This function generates the anomaly flag and and probability for the scoring window.
        :param pandas.DataFrame df: Input training data frame.
        :param str imputed_metric: Column storing the time series values.
        :param float anomaly_scores_mean: Mean of the anomaly scores for the traing sub-windows.
        :param float anomaly_scores_sd: Standard deviation of the anomaly scores for the traing sub-windows.
        :param list baseline: A list storing a baseline window used to score the scoring window.
        :param int detrend_order: The order of detrending based on MA or differencing method.
        :param str detrend_method: Selects between "ma" or "diff" detrend method.
        :param float ma_forecast_adj: Adjustment for the forecasting window in case MA based detrending applied.
        :param attributes: Model attributes.
        :param list training_tail: Last training sub-window.
        :return: Returns the probability of anomaly with the corresponding anomaly probability.
        :rtype: tuple(bool, float, dict)
        """

        import pandas as pd

        is_anomaly, prob_of_anomaly = self._anomalous_region_detection(input_df=df, value_column=imputed_metric,
                                                                       called_for="scoring",
                                                                       anomaly_scores_mean=anomaly_scores_mean,
                                                                       anomaly_scores_sd=anomaly_scores_sd,
                                                                       baseline=baseline,
                                                                       detrend_order=detrend_order,
                                                                       detrend_method=detrend_method,
                                                                       ma_forecast_adj=ma_forecast_adj,
                                                                       training_tail=training_tail)

        return is_anomaly, prob_of_anomaly, attributes

    def _get_result(self, input_df=None, detrend_order=None, ma_forecast_adj=None, value_column=None,
                 ma_window_length=None, detrend_method=None, baseline_type=None, detection_method=None,
                 baseline=None, anomaly_scores_mean=None, anomaly_scores_sd=None, training_tail=None):
        """
        The function scores the scoring window for anomalies based on the training metrics and the baseline
        :param pandas.DataFrame input_df: Input data containing the training and the scoring data.
        :param list scoring_window: A list containing the start and the end of the scoring window.
        :param int detrend_order: The order of detrending based on MA or differencing method.
        :param ma_forecast_adj: Adjustment for the forecasting window in case MA based detrending applied.
        :param str value_column: Column containing the values.
        :param int ma_window_length: Length of the moving average window to be used for detrending.
        :param str detrend_method: Selects between "ma" or "diff" detrend method.
        :param str baseline_type: Selects between "aggregated" or "last_window" baseline.
        :param str detection_method: Selects between "kldiv" or "sign_test" distance method.
        :param list baseline: A list storing a baseline window used to score the scoring window.
        :param float anomaly_scores_mean: Mean of the anomaly scores between training sub-windows.
        :param float anomaly_scores_sd: Standard deviation of the anomaly scores between training sub-windows.
        :param float significance_level: Significance level for anomaly detection
        :param list training_tail: Last training sub-window.
        :return: Returns the probability of anomaly with the corresponding anomaly probability.
        :rtype: tuple(bool, float)
        """

        import numpy as np
        import pandas as pd
        import scipy.stats as st

        is_anomaly = False
        execution_data = input_df[value_column]

        if detrend_method == 'diff':
            # Obtain the execution data and perform the necessary differencing
            execution_data = list(execution_data)
            execution_data = np.diff(execution_data, detrend_order).tolist() if detrend_order > 0 \
                else execution_data
        elif detrend_method == 'ma':
            if detrend_order > 0:
                execution_data = execution_data.to_list()
                mock_ma_window_left = np.array(training_tail[-int(ma_window_length / 2.0):]) * ma_forecast_adj
                mock_ma_window_right = np.array(training_tail[:int(ma_window_length / 2.0)]) * ma_forecast_adj
                ma_execution_data = list(mock_ma_window_left) + execution_data + list(mock_ma_window_right)

                de_obj = DataExploration()
                execution_data = de_obj._ma_detrender(series=execution_data, padded_series=ma_execution_data,
                                                      ma_window_length=ma_window_length)
            else:
                execution_data = list(execution_data)

        # Kl divergence based anomaly detection
        if detection_method == "kldiv":
            current_anomaly_score = self._distance_function(data=execution_data,
                                                            called_for="scoring", baseline=baseline)
            if current_anomaly_score > st.norm.ppf(1 - self.sig_level, anomaly_scores_mean, anomaly_scores_sd):
                is_anomaly = True

            prob_of_anomaly = st.norm.cdf(current_anomaly_score, anomaly_scores_mean, anomaly_scores_sd)
        # Sign test based anomaly detection
        elif detection_method == "sign_test":
            # If last window is the baseline, we perform the Wilcoxon sign rank test for means and levene
            # test for variance to detect anomalies
            if baseline_type == "last_window":
                test_stat_wilcoxon, pvalue_wilcoxon = st.wilcoxon(execution_data, baseline)
                test_stat_levene, pvalue_levene = st.levene(execution_data, baseline)
                if pvalue_wilcoxon < self.sig_level or pvalue_levene < self.sig_level:
                    is_anomaly = True
                prob_of_anomaly = 1 - min(pvalue_wilcoxon, pvalue_levene)
            # If aggregated is the baseline, we perform the Wilcoxon sign rank test for means and gamma distribution
            # based test for the past standard deviations to detect anomalies
            elif baseline_type == "aggregated":
                baseline_mean = np.array(baseline).mean(0).tolist()
                baseline_sds = np.array(baseline).std(1).tolist()
                test_stat_wilcoxon, pvalue_wilcoxon = st.wilcoxon(execution_data, baseline_mean)
                gamma_alpha, gamma_loc, gamma_beta = st.gamma.fit(baseline_sds)
                pvalue_gamma = 1 - st.gamma.cdf(np.std(execution_data), gamma_alpha, gamma_loc, gamma_beta)
                if pvalue_wilcoxon < self.sig_level or pvalue_gamma < self.sig_level:
                    is_anomaly = True
                prob_of_anomaly = 1 - min(pvalue_wilcoxon, pvalue_gamma)

        return is_anomaly, prob_of_anomaly

    def score(self, data, **kwargs):
        """
        Function scores input series for anomalies

        :param pandas.DataFrame data: Input time series to score
        :return: Output dictionary with scoring summary.
        :rtype: dict

        >>> data
                                raw interpolated
        index
        2018-10-06 00:00:00  204800       204800
        2018-10-06 01:00:00  222218       222218
        2018-10-06 02:00:00  218903       218903
        2018-10-06 03:00:00  190639       190639
        2018-10-06 04:00:00  148214       148214
        2018-10-06 05:00:00  106358       106358
        2018-10-06 06:00:00   70081        70081
        2018-10-06 07:00:00   47748        47748
        2018-10-06 08:00:00   36837        36837
        2018-10-06 09:00:00   33023        33023
        2018-10-06 10:00:00   44432        44432
        2018-10-06 11:00:00   72773        72773
        2018-10-06 12:00:00  115180       115180
        2018-10-06 13:00:00  157568       157568
        2018-10-06 14:00:00  180174       180174
        2018-10-06 15:00:00  190048       190048
        2018-10-06 16:00:00  188391       188391
        2018-10-06 17:00:00  189233       189233
        2018-10-06 18:00:00  191703       191703
        2018-10-06 19:00:00  189848       189848
        2018-10-06 20:00:00  192685       192685
        2018-10-06 21:00:00  196743       196743
        2018-10-06 22:00:00  193016       193016
        2018-10-06 23:00:00  196441       196441
        >>> model
        <luminaire.model.window_density.WindowDensityModel object at 0x7fcaab72fdd8>

        >>> model.score(data)
        {'Success': True, 'ConfLevel': 99.9, 'IsAnomaly': False, 'AnomalyProbability': 0.6963188902776808}
        """

        import numpy as np

        freq = self._params['freq']

        is_log_transformed = self._params['is_log_transformed']
        detrend_method = self._params['detrend_method']
        target_metric = 'raw'
        imputed_metric = 'interpolated'
        if freq not in ['S', 'M', 'QM', 'H', 'D']:
            detection_method = self._params['detection_method']
            if not detection_method:
                raise ValueError('Detection method should be specified in case frequency not in the specified list')
            if detrend_method == 'ma':
                ma_window_length = self._params['ma_window_length']
                if not ma_window_length:
                    raise ValueError(
                        'Moving average window length should be specified for detrending for frequency not in the '
                        'specified list')

        # We want to make sure the time series does not contain any negatives in case of log transformation
        if is_log_transformed:
            neg_flag = True if not data[data[target_metric] < 0].empty else False
            data[imputed_metric] = data[imputed_metric] if neg_flag else np.log(data[imputed_metric] + 1)

        anomaly_scores_mean = self._params['AnomalyScoresMean']
        anomaly_scores_sd = self._params['AnomalyScoresSD']
        baseline = self._params['Baseline']
        detrend_order = self._params['NonStationarityOrder']
        ma_forecast_adj = self._params['MovAvgForecastAdj']

        is_anomaly, prob_of_anomaly, attributes = self._call_scoring(df=data,
                                                                     imputed_metric=imputed_metric,
                                                                     anomaly_scores_mean=anomaly_scores_mean,
                                                                     anomaly_scores_sd=anomaly_scores_sd,
                                                                     baseline=baseline,
                                                                     detrend_order=detrend_order,
                                                                     detrend_method=detrend_method,
                                                                     ma_forecast_adj=ma_forecast_adj,
                                                                     training_tail=self._params['TrainingTail'])

        result = {'Success': True,
                  'ConfLevel': float(1.0 - self.sig_level) * 100,
                  'IsAnomaly': is_anomaly,
                  'AnomalyProbability': float(prob_of_anomaly),
                  }

        return result


    def _anomalous_region_detection(self, input_df=None, window_length=None,
                                   training_window=None,
                                   ignore_window=None, value_column=None, called_for=None,
                                   anomaly_scores_mean=None,
                                   anomaly_scores_sd=None, detrend_order=None, baseline=None, detrend_method=None,
                                   ma_forecast_adj=None, training_tail=None):
        """
        This function detects anomaly given a training and a scoring window.

        :param pandas.DataFrame input_df: Input data containing the training and the scoring data.
        :param int window_length: The length of a training sub-window / scoring window.
        :param list training_window: A list containing the start and the end of the training window.
        :param list scoring_window: A list containing the start and the end of the training window.
        :param int ignore_window: ignore a time window to be considered for training.
        :param str value_column: A string identifying the value column from the input dataframe
        :param str called_for: A flag to specify whether this function is called for training or scoring.
        :param float significance_level: The significance level to use when determining anomalies. This should be a
        number between 0 and 1, with values closer to 1 generating more anomalies.
        :param float anomaly_scores_mean: Means of the past anomaly scores.
        :param float anomaly_scores_sd: Standard deviation of the past anomaly scores.
        :param int detrend_order: Number of differencing for the scoring data. Only required if called for scoring.
        :param list baseline: The baseline for the scoring. only required if called for scoring.
        :return: Anomaly flag with the corresponding probability of anomaly.
        :rtype: tuple(bool, float)

        """

        ma_window_length = self._params['ma_window_length']
        detection_method = self._params['detection_method']
        baseline_type = self._params['baseline_type']

        input_df.fillna(0, inplace=True)

        # The function can be called for either training or scoring
        if called_for == "training":

            return self._get_model(input_df=input_df,
                                  training_window=training_window,
                                  window_length=window_length,
                                  ignore_window=ignore_window,
                                  value_column=value_column,
                                  ma_window_length=ma_window_length,
                                  detrend_method=detrend_method,
                                  baseline_type=baseline_type,
                                  detection_method=detection_method)

        elif called_for == "scoring":

            return self._get_result(input_df=input_df,
                                    detrend_order=detrend_order,
                                    ma_forecast_adj=ma_forecast_adj,
                                    value_column=value_column,
                                    ma_window_length=ma_window_length,
                                    detrend_method=detrend_method,
                                    baseline_type=baseline_type,
                                    detection_method=detection_method,
                                    baseline=baseline,
                                    anomaly_scores_mean=anomaly_scores_mean,
                                    anomaly_scores_sd=anomaly_scores_sd,
                                    training_tail=training_tail)