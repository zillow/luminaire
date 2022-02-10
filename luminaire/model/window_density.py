from luminaire.model.base_model import BaseModel, BaseModelHyperParams
from luminaire.exploration.data_exploration import DataExploration


class WindowDensityHyperParams(BaseModelHyperParams):
    """
    Hyperparameter class for Luminaire Window density model.

    :param str freq: The frequency of the time-series. Luminaire supports default configuration for 'S', T, '15T',
        'H', 'D'. Any other frequency type should be specified as 'custom' and configuration should be set manually.
    :param float max_missing_train_prop: Maximum proportion of missing observation allowed in the training data.
    :param bool is_log_transformed: A flag to specify whether to take a log transform of the input data.
        If the data contain negatives, is_log_transformed is ignored even though it is set to True.
    :param str baseline_type: A string flag to specify whether to take set a baseline as the previous sub-window from
        the training data for scoring or to aggregate the overall window as a baseline. Possible values:

        - "last_window"
        - "aggregated"

    :param str detection_method: A string that select between two window testing method. Possible values:

        - "kldiv" (KL-divergence). This is recommended to be set for high frequency time series such as 'S', 'T' etc.
        - "sign_test" (Wilcoxon sign rank test). This is recommended to be set for low frequency time series such as 'H', 'D' etc.

    :param int min_window_length: Minimum size of the scoring window / a stable training sub-window length.
    
        .. Note :: This is not the minimum size of the whole training window which is the combination of stable sub-windows.

    :param int max_window_length: Maximum size of the scoring window / a stable training sub-window length.
    
        .. Note :: This is not the maximum size of the whole training window which is the combination of stable sub-windows.

    :param int window_length: Size of the scoring window / a stable training sub-window length.
    
        .. Note :: This is not the size of the whole training window which is the combination of stable sub-windows.
    
    :param str detrend_method: A string that select between two stationarizing method. Possible values:

        - "ma" (moving average based)
        - "diff" (differencing based).
    """
    def __init__(self,
                 freq=None,
                 max_missing_train_prop=0.1,
                 is_log_transformed=False,
                 baseline_type="aggregated",
                 detection_method=None,
                 min_window_length=None,
                 max_window_length=None,
                 window_length=None,
                 detrend_method='modeling'
                 ):

        super(WindowDensityHyperParams, self).__init__(
            model_name="WindowDensityModel",
            freq=freq,
            max_missing_train_prop=max_missing_train_prop,
            is_log_transformed=is_log_transformed,
            baseline_type=baseline_type,
            detection_method=detection_method,
            min_window_length=min_window_length,
            max_window_length=max_window_length,
            window_length=window_length,
            detrend_method=detrend_method
        )


class WindowDensityModel(BaseModel):
    """
    This model detects anomalous windows using KL divergence (for high frequency data) and Wilcoxon sign rank test
    (for low frequency data). This default monitoring frequency is set to pandas time frequency type 'T'.

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
        from bayesian_changepoint_detection.priors import const_prior
        from bayesian_changepoint_detection.bayesian_models import offline_changepoint_detection
        import bayesian_changepoint_detection.offline_likelihoods as offline_ll
        from functools import partial

        # Volume shift detection over the means of the training window
        q, p, pcp = offline_changepoint_detection(
            data=np.array(mean_list),
            prior_function=partial(const_prior, p=1/(len(mean_list) + 1)),
            log_likelihood_class=offline_ll.StudentT(),
            truncate=-10)

        mask_mean = np.append(0, np.exp(pcp).sum(0)) > probability_threshold

        # Volume shift detection over the standard deviations of the training window
        change_points = np.array(mask_mean).nonzero()
        last_mean_cp = change_points[0][-1] if len(change_points[0]) > 0 else []

        q, p, pcp = offline_changepoint_detection(
            data=np.array(sd_list),
            prior_function=partial(const_prior, p=1/(len(sd_list) + 1)),
            log_likelihood_class=offline_ll.StudentT(),
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
                q = stats.gaussian_kde(data[i])
                p = stats.gaussian_kde(data[i + 1])

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
            q = stats.gaussian_kde(baseline)
            p = stats.gaussian_kde(data)

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

    def _call_training(self, df=None, window_length=None, imputed_metric=None, detrend_method=None,
                       detection_method=None, freq=None, **kwargs):
        """
        This function generates the baseline and training metrics to be used for scoring.

        :param pandas.DataFrame df: Input training data frame.
        :param int window_length: The length of a training sub-window.
        :param str imputed_metric: Column storing the time series values.
        :param str detrend_method: Detrend method "modeling" or "diff" for nonstationarity.
        :param str detection_method: Detection method "kldiv" or "sign_test".
        :param str freq: Data frequency.
        :return: Returns past anomaly scores based on training data, baseline and other related metrics.
        :rtype: tuple(list, float, float, float, int, list, luminaire.model, float, dict, list)
        """
        import pandas as pd

        past_anomaly_scores = dict()
        gamma_alpha = dict()
        gama_loc = dict()
        gamma_beta = dict()
        detrend_order = dict()
        baseline = dict()
        agg_data_model = dict()
        agg_data = dict()

        past_model = kwargs.get('past_model')
        training_start = df.first_valid_index()
        training_end = df.last_valid_index()
        current_training_end = training_end

        while (training_end - current_training_end) < pd.Timedelta('1D'):
            df_current = df[df.index <= current_training_end]
            past_anomaly_scores_current, gamma_alpha_current, gama_loc_current, gamma_beta_current, \
            detrend_order_current, baseline_current, agg_data_model_current, \
            agg_data_current = self._anomalous_region_detection(input_df=df_current,
                                                                window_length=window_length,
                                                                value_column=imputed_metric,
                                                                called_for="training",
                                                                detrend_method=detrend_method,
                                                                past_model=past_model,
                                                                detection_method=detection_method)

            past_anomaly_scores.update({str(current_training_end.time().strftime('%H:%M:%S')): past_anomaly_scores_current})
            gamma_alpha.update({str(current_training_end.time().strftime('%H:%M:%S')): float(gamma_alpha_current) if gamma_alpha_current else None})
            gama_loc.update({str(current_training_end.time().strftime('%H:%M:%S')): float(gama_loc_current) if gama_loc_current else None})
            gamma_beta.update({str(current_training_end.time().strftime('%H:%M:%S')): float(gamma_beta_current) if gamma_beta_current else None})
            detrend_order.update({str(current_training_end.time().strftime('%H:%M:%S')): detrend_order_current})
            baseline.update({str(current_training_end.time().strftime('%H:%M:%S')): baseline_current})
            agg_data_model.update({str(current_training_end.time().strftime('%H:%M:%S')): agg_data_model_current})
            agg_data.update({str(current_training_end.time().strftime('%H:%M:%S')): agg_data_current})

            if isinstance(freq, str):
                freq = pd.Timedelta('1' + freq)
            current_training_end = current_training_end - min(pd.Timedelta('30T'), freq * 10)

        return past_anomaly_scores, gamma_alpha, gama_loc, gamma_beta, \
               detrend_order, baseline, agg_data_model, agg_data, training_start, training_end


    def _get_model(self, input_df=None, window_length=None, value_column=None, detrend_method=None, baseline_type=None,
                   detection_method=None, past_model=None):
        """
        This function runs the training process given the input parameters.
        :param pandas.DataFrame input_df: Input data containing the training and the scoring data.
        :param int window_length: The length of a training sub-window / scoring window.
        :param str value_column: Column containing the values.
        :param str detrend_method: Selects between "modeling" or "diff" detrend method.
        :param str baseline_type: Selects between "aggregated" or "last_window" baseline.
        :param str detection_method: Selects between "kldiv" or "sign_test" distance method.
        :param luminaire.model.window_density.WindowDensityModel past_model: luminaire.model to append model metadata from past
        :return: Returns past anomaly scores based on training data, baseline and other related metrics.
        :rtype: tuple(list, float, float, float, int, list, luminaire.model, float)
        """
        import numpy as np
        import pandas as pd
        from itertools import chain
        import scipy.stats as st

        model_history_truncation_prop = 0.25    # This is the proportion of history to truncate from both sides
        # everytime we store the past anomaly scores

        de_obj = DataExploration()
        sliced_training_data, agg_datetime = de_obj._partition(input_df, window_length, value_column)

        # performing the stationarity test
        sliced_training_data_cleaned, detrend_order, agg_data_model, agg_data = de_obj._detrender(
            training_data_sliced=sliced_training_data,
            significance_level=0.05,
            detrend_method=detrend_method,
            agg_datetime=agg_datetime,
            past_model=past_model)

        # Obtain the past anomaly scores and the anomaly means and standard deviation if the detection method
        # is KL divergence
        if detection_method == "kldiv":
            past_anomaly_scores = np.array(self._distance_function(data=sliced_training_data_cleaned,
                                                                   called_for="training"))

            if past_model:
                model_timestamps = list(past_model._params['PastAnomalyScores'].keys())
                training_end = input_df.index[-1]
                current_min_timedelta = pd.Timedelta('10D')
                for timestamp in model_timestamps:
                    current_datetime = pd.Timestamp(str(training_end.date()) + ' ' + timestamp)
                    temp_timedelta = training_end - current_datetime
                    temp_timedelta = pd.Timedelta('1D') + temp_timedelta if temp_timedelta < pd.Timedelta(
                        0) else temp_timedelta
                    if temp_timedelta < current_min_timedelta:
                        opt_timestamp = timestamp
                        current_min_timedelta = temp_timedelta

                past_anomaly_scores = np.concatenate([past_model._params['PastAnomalyScores'][opt_timestamp][
                                                      int(len(past_anomaly_scores) * model_history_truncation_prop):
                                                      -int(len(past_anomaly_scores) * model_history_truncation_prop)]
                                                         , past_anomaly_scores])

            if len(past_anomaly_scores) < 100:
                alpha = []
                loc = []
                beta = []
                for i in range(10):
                    boot_scores = np.random.choice(past_anomaly_scores.tolist(), size=100, replace=True)
                    alpha_i, loc_i, beta_i = st.gamma.fit(boot_scores)
                    alpha.append(alpha_i)
                    loc.append(loc_i)
                    beta.append(beta_i)
                gamma_alpha = np.mean(alpha)
                gamma_loc = np.mean(loc)
                gamma_beta = np.mean(beta)
            else:
                gamma_alpha, gamma_loc, gamma_beta = st.gamma.fit(past_anomaly_scores)
        else:
            past_anomaly_scores, gamma_alpha, gamma_loc, gamma_beta = None, None, None, None

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

        return past_anomaly_scores, gamma_alpha, gamma_loc, gamma_beta, detrend_order, \
               baseline, agg_data_model, agg_data

    def train(self, data, **kwargs):
        """
        Input time series for training.

        :param pandas.DataFrame data: Input time series.
        :return: Trained model with the training timestamp and a success flag
        :rtype: tuple(bool, str, python model object)

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
        (True, "2018-10-10 23:00:00", <luminaire.model.window_density.WindowDensityModel object at 0x7fd7c5a34e80>)
        """
        import numpy as np
        import pandas as pd

        freq = pd.Timedelta(self._params['freq']) if self._params['freq'] not in ['S', 'T', '15T', 'H', 'D'] \
            else self._params['freq']
        if freq in ['S', 'T', '15T', 'H', 'D']:
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
        detrend_method = self._params['detrend_method']
        target_metric = 'raw'
        imputed_metric = 'interpolated'
        if not self._params['detection_method']:
            if freq in ['S', 'T', '15T']:
                detection_method = 'kldiv'
            elif freq in ['H', 'D']:
                detection_method = 'sign_test'
            else:
                detection_method = 'sign_test' if freq > np.timedelta64(30, 'm') else 'kldiv'
        else:
            detection_method = self._params['detection_method']

        if len(data) == 0:
            model = {'ErrorMessage': 'DataFrame length is 0'}
            success = False
            return success, WindowDensityModel(**model)

        # Shift the interpolated value by +1 and get the log. This handles values with 0.
        if is_log_transformed:
            neg_flag = True if not data[data[target_metric] < 0].empty else False
            data[imputed_metric] = data[imputed_metric] if neg_flag else np.log(data[imputed_metric] + 1)

        past_anomaly_scores, anomaly_scores_gamma_alpha, anomaly_scores_gamma_loc, anomaly_scores_gamma_beta, \
        detrend_order, baseline, agg_data_model, agg_data, \
        training_start, training_end = self._call_training(df=data, window_length=window_length,
                                                           imputed_metric=imputed_metric,
                                                           detrend_method=detrend_method,
                                                           detection_method=detection_method,
                                                           freq=freq, **kwargs)

        success = True
        self.hyper_params['is_log_transformed'] = is_log_transformed
        self.hyper_params['detection_method'] = detection_method
        model = {'TrainingStartDate': str(training_start),
                 'PastAnomalyScores': past_anomaly_scores,
                 'AnomalyScoresGammaAlpha': anomaly_scores_gamma_alpha,
                 'AnomalyScoresGammaLoc': anomaly_scores_gamma_loc,
                 'AnomalyScoresGammaBeta': anomaly_scores_gamma_beta,
                 'NonStationarityOrder': detrend_order,
                 'Baseline': baseline,
                 'AggregatedDataModel': agg_data_model,
                 'AggregatedData': agg_data
                 }

        return success, str(training_end), WindowDensityModel(hyper_params=self.hyper_params, **model)

    def _call_scoring(self, df=None, target_metric=None, anomaly_scores_gamma_alpha=None, anomaly_scores_gamma_loc=None,
                      anomaly_scores_gamma_beta=None, baseline=None, detrend_order=None, detrend_method=None,
                      agg_data_model=None, detection_method=None, attributes=None, agg_data=None):
        """
        This function generates the anomaly flag and and probability for the scoring window.
        :param pandas.DataFrame df: Input training data frame.
        :param str target_metric: Column storing the time series values.
        :param float anomaly_scores_gamma_alpha: Gamma fit alpha parameter.
        :param float anomaly_scores_gamma_loc: Gamma fit location parameter.
        :param float anomaly_scores_gamma_beta: Gamma fit beta parameter.
        :param list baseline: A list storing a baseline window used to score the scoring window.
        :param int detrend_order: The order of detrending based on MA or differencing method.
        :param str detrend_method: Selects between "modeling" or "diff" detrend method.
        :param luminaire.model.lad_structural.LADStructuralModel agg_data_model: Prediction model for aggregated data.
        :param str detection_method: Selects between "kldiv" or "sign_test" distance method.
        :param attributes: Model attributes.
        :param agg_data: Aggregated Data per day.
        :return: Returns the anomaly flag with the corresponding anomaly probability.
        :rtype: tuple(bool, float, dict)
        """

        is_anomaly, prob_of_anomaly = self._anomalous_region_detection(input_df=df, value_column=target_metric,
                                                                       called_for="scoring",
                                                                       anomaly_scores_gamma_alpha=anomaly_scores_gamma_alpha,
                                                                       anomaly_scores_gamma_loc=anomaly_scores_gamma_loc,
                                                                       anomaly_scores_gamma_beta=anomaly_scores_gamma_beta,
                                                                       baseline=baseline,
                                                                       detrend_order=detrend_order,
                                                                       detrend_method=detrend_method,
                                                                       agg_data_model=agg_data_model,
                                                                       detection_method=detection_method,
                                                                       agg_data=agg_data)

        return is_anomaly, prob_of_anomaly, attributes

    def _get_result(self, input_df=None, detrend_order=None, agg_data_model=None, value_column=None,
                    detrend_method=None, baseline_type=None, detection_method=None, baseline=None,
                    anomaly_scores_gamma_alpha=None, anomaly_scores_gamma_loc=None, anomaly_scores_gamma_beta=None,
                    agg_data=None):
        """
        The function scores the scoring window for anomalies based on the training metrics and the baseline
        :param pandas.DataFrame input_df: Input data containing the training and the scoring data.
        :param int detrend_order: The non-negative order of detrending based on Modeling or differencing method. When
        the detrend_order > 0, corresponding detrending need to be performed using the method specified in the model
        config.
        :param luminaire.model.lad_structural.LADStructuralModel agg_data_model: Prediction model for aggregated data.
        :param str value_column: Column containing the values.
        :param str detrend_method: Selects between "modeling" or "diff" detrend method.
        :param str baseline_type: Selects between "aggregated" or "last_window" baseline.
        :param str detection_method: Selects between "kldiv" or "sign_test" distance method.
        :param list baseline: A list storing a baseline window used to score the scoring window.
        :param float anomaly_scores_gamma_alpha: Gamma fit alpha parameter.
        :param float anomaly_scores_gamma_loc: Gamma fit location parameter.
        :param float anomaly_scores_gamma_beta: Gamma fit beta parameter.
        :param agg_data: Aggregated Data per day.
        :return: Returns the anomaly flag with the corresponding anomaly probability.
        :rtype: tuple(bool, float)
        """

        import numpy as np
        import pandas as pd
        import copy
        import scipy.stats as st
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.covariance import EmpiricalCovariance, MinCovDet
        import collections
        import operator

        is_anomaly = False
        execution_data = input_df[value_column]
        adjusted_execution_data = []
        prob_of_anomaly = []
        len_req_agg_data_model = 42     # Setting a hard threshold to have predictions from aggregated data
        # for stationarity adjustment

        if detrend_method == 'diff':
            # Obtain the execution data and perform the necessary differencing
            execution_data = list(execution_data)
            adjusted_execution_data = np.diff(execution_data, detrend_order).tolist() if detrend_order > 0 \
                else execution_data
        elif detrend_method == 'modeling':
            idx = input_df.index.normalize()
            dates_freq_dist = dict(collections.Counter(idx))
            scoring_datetime = str(max(dates_freq_dist.items(), key=operator.itemgetter(1))[0])
            execution_data_avg = np.mean(execution_data)
            # If detrending is needed, we scale the scoring data accordingly using the agg_dat_model forecast
            if detrend_order > 0:
                snapshot_len_max = min(len(agg_data), len_req_agg_data_model)
                agg_data_trunc = np.array(agg_data)[:, 1][-snapshot_len_max:]
                data_adjust_forecast = []
                try:
                    # Setting the data adjustment window of the original data using the predictions and the CILower and
                    # CIUpper keeping the prediction uncertainty of the agg_model in mind
                    if agg_data_model and len(agg_data) > len_req_agg_data_model:
                        score = agg_data_model.score(execution_data_avg, scoring_datetime)
                        data_adjust_forecast.append(score['Prediction'])
                        data_adjust_forecast.append(score['CILower'])
                        data_adjust_forecast.append(score['CIUpper'])
                    else:
                        data_adjust_forecast.append(np.median(agg_data_trunc))
                        data_adjust_forecast.append(np.percentile(agg_data_trunc, 5))       # setting a 2-sigma limit
                        data_adjust_forecast.append(np.percentile(agg_data_trunc, 95))       # setting a 2-sigma limit
                except:
                    # If the scoring for the agg_data_model fails for some reason, we use the latest agg_data for the
                    # detrending adjustment
                    data_adjust_forecast.append(np.median(agg_data_trunc))
                    data_adjust_forecast.append(np.percentile(agg_data_trunc, 5))       # setting a 2-sigma limit
                    data_adjust_forecast.append(np.percentile(agg_data_trunc, 95))       # setting a 2-sigma limit
                for i in range(3):
                    if data_adjust_forecast[i] != 0:
                        adjusted_execution_data.append((execution_data / data_adjust_forecast[i]).tolist())
            else:
                adjusted_execution_data = list(execution_data)

        # Kl divergence based anomaly detection
        if detection_method == "kldiv":
            if detrend_order > 0:
                prob_of_anomaly = []
                for i in range(3):
                    current_anomaly_score = self._distance_function(data=adjusted_execution_data[i],
                                                                    called_for="scoring", baseline=baseline)
                    prob_of_anomaly.append(st.gamma.cdf(current_anomaly_score, anomaly_scores_gamma_alpha,
                                                        anomaly_scores_gamma_loc, anomaly_scores_gamma_beta))
                prob_of_anomaly = np.min(prob_of_anomaly)
            else:
                current_anomaly_score = self._distance_function(data=adjusted_execution_data,
                                                                called_for="scoring", baseline=baseline)
                prob_of_anomaly = st.gamma.cdf(current_anomaly_score, anomaly_scores_gamma_alpha,
                                               anomaly_scores_gamma_loc, anomaly_scores_gamma_beta)

            if 1 - prob_of_anomaly < self.sig_level:
                is_anomaly = True
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
                baseline_sds = np.array(baseline).std(1).tolist()
                if detrend_order == 0:
                    # crearing a 2d list to make it easy to loop through in the following for loop
                    adjusted_execution_data = [adjusted_execution_data]
                for current_adjusted_data in adjusted_execution_data:
                    baseline_execution_data = copy.copy(baseline)
                    baseline_execution_data.append(current_adjusted_data)
                    pca = PCA()
                    scores = pca.fit_transform(StandardScaler().fit_transform(baseline_execution_data))
                    robust_cov = MinCovDet().fit(scores[:, :3])
                    mahalanobis_distance = robust_cov.mahalanobis(scores[:, :3])        # getting the top 3 dimensions
                    pvalue_mahalanobis = 1 - st.chi2.cdf(mahalanobis_distance[-1],
                                                         np.array(baseline_execution_data).shape[1])

                    gamma_alpha, gamma_loc, gamma_beta = st.gamma.fit(baseline_sds)
                    pvalue_gamma = 1 - st.gamma.cdf(np.std(current_adjusted_data), gamma_alpha, gamma_loc, gamma_beta)
                    if pvalue_mahalanobis < self.sig_level or pvalue_gamma < self.sig_level:
                        is_anomaly = True
                    prob_of_anomaly.append(1 - min(pvalue_mahalanobis, pvalue_gamma))
                prob_of_anomaly = np.min(prob_of_anomaly)

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
        2018-10-11 00:00:00  204800       204800
        2018-10-11 01:00:00  222218       222218
        2018-10-11 02:00:00  218903       218903
        2018-10-11 03:00:00  190639       190639
        2018-10-11 04:00:00  148214       148214
        2018-10-11 05:00:00  106358       106358
        2018-10-11 06:00:00   70081        70081
        2018-10-11 07:00:00   47748        47748
        2018-10-11 08:00:00   36837        36837
        2018-10-11 09:00:00   33023        33023
        2018-10-11 10:00:00   44432        44432
        2018-10-11 11:00:00   72773        72773
        2018-10-11 12:00:00  115180       115180
        2018-10-11 13:00:00  157568       157568
        2018-10-11 14:00:00  180174       180174
        2018-10-11 15:00:00  190048       190048
        2018-10-11 16:00:00  188391       188391
        2018-10-11 17:00:00  189233       189233
        2018-10-11 18:00:00  191703       191703
        2018-10-11 19:00:00  189848       189848
        2018-10-11 20:00:00  192685       192685
        2018-10-11 21:00:00  196743       196743
        2018-10-11 22:00:00  193016       193016
        2018-10-11 23:00:00  196441       196441
        >>> model
        <luminaire.model.window_density.WindowDensityModel object at 0x7fcaab72fdd8>

        >>> model.score(data)
        {'Success': True, 'ConfLevel': 99.9, 'IsAnomaly': False, 'AnomalyProbability': 0.6963188902776808}
        """

        import numpy as np
        import pandas as pd

        is_log_transformed = self._params['is_log_transformed']
        detrend_method = self._params['detrend_method']
        target_metric = 'raw'
        imputed_metric = 'interpolated'
        detection_method = self._params['detection_method']

        # We want to make sure the time series does not contain any negatives in case of log transformation
        if is_log_transformed:
            neg_flag = True if not data[data[target_metric] < 0].empty else False
            data[imputed_metric] = data[imputed_metric] if neg_flag else np.log(data[imputed_metric] + 1)

        model_timestamps = list(self._params['AnomalyScoresGammaAlpha'].keys())
        scoring_start = data.index[0]
        current_min_timedelta = pd.Timedelta('10D')
        for timestamp in model_timestamps:
            current_datetime = pd.Timestamp(str(scoring_start.date()) + ' ' + timestamp)
            temp_timedelta = scoring_start - current_datetime
            temp_timedelta = pd.Timedelta('1D') + temp_timedelta if temp_timedelta < pd.Timedelta(0) else temp_timedelta
            if temp_timedelta < current_min_timedelta:
                opt_timestamp = timestamp
                current_min_timedelta = temp_timedelta

        anomaly_scores_gamma_alpha = self._params['AnomalyScoresGammaAlpha'][opt_timestamp]
        anomaly_scores_gamma_loc = self._params['AnomalyScoresGammaLoc'][opt_timestamp]
        anomaly_scores_gamma_beta = self._params['AnomalyScoresGammaBeta'][opt_timestamp]
        baseline = self._params['Baseline'][opt_timestamp]
        detrend_order = self._params['NonStationarityOrder'][opt_timestamp]
        agg_data_model = self._params['AggregatedDataModel'][opt_timestamp]
        agg_data = self._params['AggregatedData'][opt_timestamp]

        is_anomaly, prob_of_anomaly, attributes = self._call_scoring(df=data,
                                                                     target_metric=target_metric,
                                                                     anomaly_scores_gamma_alpha=anomaly_scores_gamma_alpha,
                                                                     anomaly_scores_gamma_loc=anomaly_scores_gamma_loc,
                                                                     anomaly_scores_gamma_beta=anomaly_scores_gamma_beta,
                                                                     baseline=baseline,
                                                                     detrend_order=detrend_order,
                                                                     detrend_method=detrend_method,
                                                                     agg_data_model=agg_data_model,
                                                                     detection_method=detection_method,
                                                                     agg_data=agg_data)

        result = {'Success': True,
                  'ConfLevel': float(1.0 - self.sig_level) * 100,
                  'IsAnomaly': is_anomaly,
                  'AnomalyProbability': float(prob_of_anomaly),
                  }

        return result, data.reset_index().values.tolist()


    def _anomalous_region_detection(self, input_df=None, window_length=None,
                                    value_column=None, called_for=None,
                                    anomaly_scores_gamma_alpha=None, anomaly_scores_gamma_loc=None,
                                    anomaly_scores_gamma_beta=None, detrend_order=None, baseline=None,
                                    detrend_method=None, agg_data_model=None, past_model=None, detection_method=None,
                                    agg_data=None):
        """
        This function detects anomaly given a training and a scoring window.

        :param pandas.DataFrame input_df: Input data containing the training and the scoring data.
        :param int window_length: The length of a training sub-window / scoring window.
        :param str value_column: A string identifying the value column from the input dataframe
        :param str called_for: A flag to specify whether this function is called for training or scoring.
        :param float anomaly_scores_gamma_alpha: Gamma fit alpha parameter.
        :param float anomaly_scores_gamma_loc: Gamma fit location parameter.
        :param float anomaly_scores_gamma_beta: Gamma fit beta parameter.
        :param int detrend_order: Number of differencing for the scoring data. Only required if called for scoring.
        :param list baseline: The baseline for the scoring. only required if called for scoring.
        :param str detrend_method: Selects between "modeling" or "diff" detrend method.
        :param luminaire.model.lad_structural.LADStructuralModel agg_data_model: Prediction model for aggregated data.
        :param luminaire.model.window_density.WindowDensityModel past_model: Past stored window density model.
        :param str detection_method: Selects between "kldiv" or "sign_test" distance method.
        :param agg_data: Aggregated Data per day.
        :return: Anomaly flag with the corresponding probability of anomaly.
        :rtype: tuple(bool, float)

        """

        baseline_type = self._params['baseline_type']

        input_df.fillna(0, inplace=True)

        # The function can be called for either training or scoring
        if called_for == "training":

            return self._get_model(input_df=input_df,
                                   window_length=window_length,
                                   value_column=value_column,
                                   detrend_method=detrend_method,
                                   baseline_type=baseline_type,
                                   detection_method=detection_method,
                                   past_model=past_model)

        elif called_for == "scoring":

            return self._get_result(input_df=input_df,
                                    detrend_order=detrend_order,
                                    agg_data_model=agg_data_model,
                                    value_column=value_column,
                                    detrend_method=detrend_method,
                                    baseline_type=baseline_type,
                                    detection_method=detection_method,
                                    baseline=baseline,
                                    anomaly_scores_gamma_alpha=anomaly_scores_gamma_alpha,
                                    anomaly_scores_gamma_loc=anomaly_scores_gamma_loc,
                                    anomaly_scores_gamma_beta=anomaly_scores_gamma_beta,
                                    agg_data=agg_data)
