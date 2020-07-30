class DataExplorationError(Exception):
    """
    Exception class for Luminaire Data Exploration.

    """
    def __init__(self, message):
        message = f'Data exploration failed. Error: {message}'
        super(DataExplorationError, self).__init__(message)


class DataExploration(object):
    """
    This is a general class for time series data exploration and pre-processing.

    :param str freq: The frequency of the time-series. A `Pandas offset`_ such as 'D', 'H', or 'M'.
    :param float sig_level: The significance level to use for any statistical test withing data profile. This should be
        a number between 0 and 1.
    :param min_ts_mean: The minimum mean value of the time series required for the model to run. For data that
        originated as integers (such as counts), the ARIMA model can behave erratically when the numbers are small. When
        this parameter is set, any time series whose mean value is less than this will automatically result in a model
        failure, rather than a mostly bogus anomaly.
    :type min_ts_mean: float, optional
    :param fill_rate: Minimum proportion of data availability in the recent data window.
    :type fill_rate: float, optional
    :param max_window_size: The maximum size of the sub windows for input data segmentation.
    :type max_window_size: int, optional
    :param window_size: The size of the sub windows for input data segmentation.
    :type window_size: int, optional
    :param min_ts_length: The minimum required length of the time series for training.
    :type min_ts_length: int, optional
    :param max_ts_length: The maximum required length of the time series for training.
    :type max_ts_length: int, optional
    :param is_log_transformed: A flag to specify whether to take a log transform of the input data. If the data
        contain negatives, is_log_transformed is ignored even though it is set to True.
    :type is_log_transformed: bool, optional
    :param data_shift_truncate: A flag to specify whether left side of the most recent change point needs to
        be truncated from the training data.
    :type data_shift_truncate: bool, optional
    :param min_changepoint_padding_length: A padding length between two change points. This parameter makes sure
        that two consecutive change points are not close to each other.
    :type min_changepoint_padding_length: bool, optional
    :param change_point_threshold: Minimum threshold (a value > 0) to flag change points based on KL divergence.
    :type change_point_threshold: float, optional

    .. _Pandas offset: https://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases
    """
    __version__ = "0.1"

    def __init__(self,
                 freq='D',
                 min_ts_mean=None,
                 fill_rate=None,
                 max_window_size=24,
                 window_size=None,
                 sig_level=0.05,
                 min_ts_length=None,
                 max_ts_length=None,
                 is_log_transformed=None,
                 data_shift_truncate=True,
                 min_changepoint_padding_length=None,
                 change_point_threshold=2,
                 *args,
                 **kwargs):

        self._target_index = 'index'
        self._target_metric = 'raw'

        # Assigning a default min_ts_length if not provided by the user. Assigning two cycles.
        # Keys are same as pandas frequency.
        min_ts_length_dict = {
            'H': 144,
            'D': 21,
            'W': 16, 'W-SUN': 16, 'W-MON': 16, 'W-TUE': 16, 'W-WED': 16, 'W-THU': 16, 'W-FRI': 16, 'W-SAT': 16,
            'M': 24, 'MS': 24,
        }
        self.min_ts_length = min_ts_length or min_ts_length_dict.get(freq)
        # Assigning a default max_ts_length if not provided by the user.
        # Keys are same as pandas frequency.
        max_ts_length_dict = {
            'H': 90 * 24,
            'D': 365 * 3,
            'W': 100, 'W-SUN': 100, 'W-MON': 100, 'W-TUE': 100, 'W-WED': 100, 'W-THU': 100, 'W-FRI': 100, 'W-SAT': 100,
            'M': 120, 'MS': 120,
        }
        self.max_ts_length = max_ts_length or max_ts_length_dict.get(freq)

        self.freq = freq
        self.min_ts_mean = min_ts_mean
        self.fill_rate = fill_rate
        self.max_window_size = max_window_size
        self.window_size = window_size
        self.sig_level = sig_level
        self.is_log_transformed = is_log_transformed
        self.data_shift_truncate = data_shift_truncate
        self.change_point_threshold = change_point_threshold
        self.min_changepoint_padding_length = min_changepoint_padding_length

        # Assigning different padding based on the time series frequency
        min_changepoint_padding_length_dict = {
            'H': 7 * 24,
            'D': 10,
            'W': 12, 'W-SUN': 12, 'W-MON': 12, 'W-TUE': 12, 'W-WED': 12, 'W-THU': 12, 'W-FRI': 12, 'W-SAT': 12,
            'M': 24, 'MS': 24,
        }
        self.min_changepoint_padding_length = min_changepoint_padding_length or min_changepoint_padding_length_dict.get(
            freq)

        window_size_dict = {
            'H': 24,
            'D': 7,
        }

        if freq in ['H', 'D']:
            self.window_size = window_size or window_size_dict.get(freq)

    def add_missing_index(self, df=None, freq=None):

        import pandas as pd

        # Create a new Pandas data frame based on the first valid index and
        # current date using the frequency defined by the use
        idx = pd.date_range(start=df.first_valid_index(), end=df.last_valid_index(), freq=freq)
        df_reindexed = df.reindex(idx)
        return df_reindexed

    def _kalman_smoothing_imputation(self, df=None, target_metric=None, imputed_metric=None, impute_only=False):
        """
        This function performs a missing data imputation using the kalman smoothing.
        :param pandas.Dataframe df: A pandas dataframe containing the time series
        :param str target_metric: A string among the dataframe column name that contains the time series
        :param str imputed_metric: A string among the dataframe column that stores the imputed time series
        :return: A pandas dataframe containing the raw and the imputed time series

        .. Note: missing data are imputed using moving average if position of the missing data does not satisfy the minimum
        length requirement for Kalman smoothing
        """
        import numpy as np
        from pykalman import KalmanFilter
        time_series = np.array(df[target_metric], dtype=np.float64)
        missing_idx = np.where(np.isnan(time_series))[0]
        if len(missing_idx) > 0:
            if missing_idx[0] == 0:
                time_series[0] = np.nanmean(time_series)
                missing_idx = np.delete(missing_idx, 0)
            if missing_idx[-1] == len(time_series) - 1:
                time_series[-1] = np.nanmean(time_series[-min(10, len(time_series)):])
                if np.isnan(time_series[-1]):
                    raise ValueError("Interpolation failed due to too many recent missing data.")
                missing_idx = np.delete(missing_idx, -1)
            if len(missing_idx) > 0:
                masked_series = np.ma.array(np.copy(time_series))
                for i in missing_idx:
                    masked_series[i] = np.ma.masked

                kf = KalmanFilter()
                ts_smoothed, ts_cov = kf.em(masked_series).smooth(masked_series)

                for i in missing_idx:
                    time_series[i] = ts_smoothed[i]

        if impute_only:
            df[target_metric] = time_series
        else:
            df[imputed_metric] = time_series

        return df

    def _moving_average(self, series=None, window_length=None, train_subwindow_len=None):
        """
        This function calculates the moving average of a series based on a given window length.

        :param list series: The list containing the training data.
        :param int window_length: The length of the moving average window.
        :return: Smoothed input series using moving averages
        :rtype: list
        """
        import numpy as np

        moving_averages = []
        iter_length = len(series) - window_length
        for i in range(0, iter_length):
            ma_current = np.mean(series[i:i + window_length])
            moving_averages.append(ma_current)

        # Moving average shrinkes w.r.t the actual series based on the moving average window. Hence, to keep the
        # length of the moving average equal to the series, we append proxy padding which are the moving averages
        # from the closest representative training sub-window.
        moving_averages_padded = moving_averages[(train_subwindow_len - (
                window_length // 2)):train_subwindow_len] + moving_averages + moving_averages[-train_subwindow_len:-(
                train_subwindow_len - (window_length // 2))]

        return moving_averages_padded

    @classmethod
    def _stationarizer(cls, endog=None, diff_min=1, diff_max=2, significance_level=0.01, obs_incl=True):
        """
        This function tests the stationarity of the given time series and performs the required differencing.

        :param pandas.Series endog: The list containing the training data. Note: endog can include the
        raw actual for specific models (e.g. State Space)
        :param int diff_min: Minimum order of differencing for non-stationarity.
        :param int diff_max: Maximum order of differencing for non-stationarity.
        :param float significance_level: Significance level for the adfuller test for checking non-stationarity
        :param bool obs_incl: A flag indicating whether the raw actual is included in the endog.
        :return: Difference time series, the order of differencing based on the stationarity test and the
        raw actuals in every differencing step (for prediction adjustment in the actual
        :rtype: tuple(numpy.array, int, list)
        """

        import numpy as np
        from statsmodels.tsa.stattools import adfuller

        endog_diff = np.array(endog)

        if diff_min == 0:
            # Performing the adfuller test over the raw and the aggregated data to test for stationarity
            adf_pvalue = adfuller(endog_diff)[1]
            take_diff_flag = adf_pvalue > significance_level
        else:
            take_diff_flag = True

        diff_order = 0

        actual_previous_per_diff = []

        # Take the difference until the difference data is stationarity based on the adfuller test
        while take_diff_flag and diff_order < diff_max:
            diff_order = diff_order + 1
            if obs_incl:
                actual_previous_per_diff.append(endog_diff[-2])
            else:
                actual_previous_per_diff.append(endog_diff[-1])
            endog_diff = np.diff(endog_diff)
            adf_pvalue = adfuller(endog_diff)[1]
            take_diff_flag = adf_pvalue > significance_level

        return endog_diff, diff_order, actual_previous_per_diff

    def _partition(self, lst, window_length):
        """
        This function slices a list from the end of the list based on the size of the slice. Any remainder part of the
        list from the beginning is ignored

        :param list lst: The list containing the training data.
        :param int window_length: The length of every slice of the list.
        :return: Sliced lists of input time series
        :rtype: list
        """
        n = int(len(lst) / float(window_length))

        # Performing pertition
        lst_sliced = [lst[::-1][int(round(window_length * i)):
                                int(round(window_length * (i + 1)))][::-1] for i in range(n)][::-1]
        return lst_sliced

    def _detrender(self, training_data_sliced=None, ma_window_length=None, detrend_order_max=2,
                   significance_level=0.01, detrend_method=None, train_subwindow_len=None):
        """
        This function tests the stationarity of the given time series and performs the required differencing.

        :param list training_data_sliced: The list of list containing the training data.
        :param int ma_window_length: The length of the moving average window.
        :param int detrend_order_max: Maximum number of differencing for non-stationarity.
        :param float significance_level: Significance level for the adfuller test for checking non-stationarity
        :return: Difference time series and the order of differencing based on the stationarity test.
        :rtype: tuple(list, int)
        """

        import numpy as np
        from itertools import chain
        from statsmodels.tsa.stattools import adfuller
        from pyramid.arima import auto_arima

        p = (0, 6)
        d = (0, 2)
        q = (0, 6)
        ma_forecast_adj = None

        # Flattening the training data using for the stationarity test
        training_data_flattened = list(chain.from_iterable(training_data_sliced))

        # Performing stationarity test over the raw and the aggregated version (means) of the training data
        if detrend_method == 'ma':
            mavges = self._moving_average(training_data_flattened, ma_window_length, train_subwindow_len)
        avg_series = np.array(training_data_sliced).mean(0).tolist()

        # Performing the adfuller test over the raw and the aggregated data to test for stationarity
        adf_pvalue_raw_series = adfuller(training_data_flattened)[1]
        adf_pvalue_avg_series = adfuller(avg_series)[1]
        detrend_flag = adf_pvalue_raw_series > significance_level or adf_pvalue_avg_series > significance_level

        detrend_order = 0

        if detrend_method == 'diff':
            training_data_sliced_stationarized = training_data_sliced
        elif detrend_method == 'ma':
            if not detrend_flag:
                return training_data_sliced, detrend_order, ma_forecast_adj
            else:
                training_data_stationarized_flattened = training_data_flattened

        # Take the difference until the difference data is stationarity based on the adfuller test
        while detrend_flag and detrend_order < detrend_order_max:
            detrend_order = detrend_order + 1
            if detrend_method == 'diff':
                for i in range(0, len(training_data_sliced)):
                    training_data_sliced_stationarized[i] = np.diff(training_data_sliced_stationarized[i])
                training_data_stationarized_flattened = list(chain.from_iterable(training_data_sliced_stationarized))

                avg_series = np.array(training_data_sliced_stationarized).mean(0).tolist()
                adf_pvalue_raw_series = adfuller(training_data_stationarized_flattened)[1]
                adf_pvalue_avg_series = adfuller(avg_series)[1]

                detrend_flag = (
                        adf_pvalue_raw_series > significance_level or adf_pvalue_avg_series > significance_level)
            elif detrend_method == 'ma':
                training_data_stationarized_flattened = (np.array(training_data_stationarized_flattened) /
                                                         np.array(mavges)).tolist()
                training_data_sliced_stationarized = self._partition(training_data_stationarized_flattened,
                                                                     train_subwindow_len)

                arima_fit = auto_arima(y=avg_series,
                                       start_p=p[0], max_p=p[1],
                                       d=d[0], max_d=d[1],
                                       start_q=q[0], max_q=q[1],
                                       seasonal=False,
                                       trace=False,
                                       method='css',
                                       solver='bfgs',
                                       error_action='ignore',  # don't want to know if an order does
                                       # not work
                                       suppress_warnings=True,  # don't want convergence warnings
                                       stepwise=True)
                model = arima_fit.arima_res_
                forecast_window_avg = model.forecast(alpha=0.05)
                ma_forecast_adj = forecast_window_avg[0] / float(avg_series[-1])
                ma_forecast_adj = ma_forecast_adj[0]

                detrend_flag = False

        return training_data_sliced_stationarized, detrend_order, ma_forecast_adj

    def _ma_detrender(self, series=None, padded_series=None, ma_window_length=None):
        """
        This function detrends a the values from a target time window w.r.t a padding around the target window.
        Note: This function is only been used for detrending the scoring window.
        :param list series: The series containing the values from the scoring window.
        :param list padded_series: The series containing the values for the padded scoring window
        :param int ma_window_length: Size of the padding
        :return: Returns 'series' after removing the trend
        :rtype: list
        """

        import numpy as np

        moving_averages = []

        iter_length = len(padded_series) - ma_window_length
        for i in range(0, iter_length):
            ma_current = np.mean(padded_series[i:i + ma_window_length])
            moving_averages.append(ma_current)

        stationarized_series = (np.array(series) / np.array(moving_averages)).tolist()

        return stationarized_series

    def _detect_window_size(self, series=None):
        """
        This function detects the ideal window size based on the seasonality pattern of the data
        :param pandas.DataFrame series: The input sequence of data.
        :param int max_window_size: The maximum size of the sub windows for input data segmentation.
        :return: An int containing the ooptimal window size
        :rtype: int
        """
        import numpy as np
        max_window_size = self.max_window_size

        n = len(series)
        series = np.diff(series, 1)
        # Generating the indices based on odd and event number of terms in the time series
        if int(n) % 2 != 0:
            all_idx = np.arange(1, n // 2 + 1)
        else:
            all_idx = np.arange(1, n // 2)
        # Performing Fourier transformation
        yf = np.real(np.fft.rfft(series))

        # Spectral density for the fourier transformation (to identify the significant frequencies)
        psd = abs(yf[all_idx]) ** 2 + abs(yf[-all_idx]) ** 2

        sig_freq_idx = np.argsort(psd)

        return min(max_window_size,
                   max(int(np.rint(float(n) // max(1, sig_freq_idx[-1]))),
                       int(np.rint(float(n) // max(1, sig_freq_idx[-2])))))

    def _local_minima(self, input_dict=None, window_size=None):
        """
        This function finds the index corresponding to the local minimas for detected consecutive trend changes
        :param dict input_dict: A dictionary containing the timestamps as keys for potential trend changes
        and the values being the corresponding p-values
        :param int window_size: The size of the sub windows for input data segmentation.
        :return: List of local minimas
        """
        import numpy as np
        import collections

        ordered_dict = collections.OrderedDict(sorted(input_dict.items()))
        key_list = np.array(list(ordered_dict.keys()))
        value_list = np.array(list(ordered_dict.values()))

        diff2_series = np.diff(np.sign(np.diff(value_list)))
        local_min_loc = np.where(diff2_series > 0)[0]
        local_min_loc = local_min_loc + 1 if len(local_min_loc) > 0 else np.zeros(1)

        if len(local_min_loc) > 1:
            min_keys = [key_list[int(loc)] for loc in local_min_loc]
        elif len(local_min_loc) <= 1:
            min_keys = [key_list[0], key_list[-1]] if len(input_dict) > window_size else [key_list[int(loc)] for loc in
                                                                                          local_min_loc]

        return min_keys

    def _shift_intensity(self, change_points=None, df=None, metric=None):
        """
        This function computes the Kullback_Leibler divergence of the the time series around a changepoint detected by the
        pelt_change_point_detection() function. This considers Gaussian assumption on the underlying data distribution.

        :param list change_points: A list storing indices of the potential change points
        :param pandas.dataframe df: A pandas dataframe containing time series ignoring the top 5% volatility
        :param str metric: A string in the dataframe column names that contains the time series
        :return: A list containing the magnitude of changes for every corresponding change points
        """
        import numpy as np

        min_changepoint_padding_length = self.min_changepoint_padding_length

        mag_change = []
        float_min = 1e-10
        c_count = 0
        # This loop iterates over all the change points obtained through PELT
        for dates in change_points:
            # KL divergence is measured for the datapoints on the left side of the change point versus the right side
            # of the change point.
            if len(change_points) == 1:
                window_l = df[metric].iloc[:change_points[c_count]].values
                window_r = df[metric].iloc[change_points[c_count]:].values
            elif c_count == 0:
                window_l = df[metric].iloc[:change_points[c_count]].values
                window_r = df[metric].iloc[change_points[c_count]:change_points[c_count + 1]].values
            elif c_count < len(change_points) - 1:
                window_l = df[metric].iloc[change_points[c_count - 1]:change_points[c_count]].values
                window_r = df[metric].iloc[change_points[c_count]:change_points[c_count + 1]].values
            else:
                window_l = df[metric].iloc[change_points[c_count - 1]:change_points[c_count]].values
                window_r = df[metric].iloc[change_points[c_count]:].values
            if len(window_r) <= min_changepoint_padding_length:
                mag_change.append(0)
            else:
                window = np.concatenate((window_l, window_r), axis=0)
                # Mean and standard deviation of the data from the combined window
                w_mean = np.mean(window)
                w_std = np.std(window, ddof=1) if len(window) > 1 else float_min
                # Mean and standard deviation of the data from the window after the change point
                wr_mean = np.mean(window_r)
                wr_std = np.std(window_r, ddof=1) if len(window_r) > 1 else float_min
                # Kullback-Leibler divergence between two normal densities
                mag_change.append(
                    np.log(wr_std / w_std) + ((w_std ** 2 + (w_mean - wr_mean) ** 2) / (2 * (wr_std ** 2))) - 0.5)
            c_count = c_count + 1

        return mag_change

    def _pelt_change_point_detection(self, df=None, metric=None, min_ts_length=None, max_ts_length=None):
        """
        This function computes the significant change points based on PELT and the Kullback-Leibler divergence method.
        :param pandas.dataframe df: A pandas dataframe containing the time series
        :param pandas.dataframe metric: The metric in the dataframe that contains the time series
        :param int min_ts_length: Specifying the minimum required length of the time series for training
        :param int max_ts_length: Specifying the maximum required length of the time series for training.
        The training time series length truncates accordingly based on minimum between max_ts_length and the
        length of the input time series.
        :return: A pandas dataframe containing the time series after the last changepoint

        >>> df
                          raw  interpolated
        2016-01-02  1753421.0     14.377080
        2016-01-03  1879108.0     14.446308
        2016-01-04  1462725.0     14.195812
        2016-01-05  1525162.0     14.237612
        2016-01-06  1424264.0     14.169166
        ...               ...           ...
        2018-10-14  2185230.0     14.597232
        2018-10-15  1825539.0     14.417386
        2018-10-16  1776778.0     14.390313
        2018-10-17  1792899.0     14.399345
        2018-10-18  1738657.0     14.368624

        >>> copy_df, change_point_list
        (                  raw  interpolated
        2016-01-02  1753421.0     14.377080
        2016-01-03  1879108.0     14.446308
        2016-01-04  1462725.0     14.195812
        2016-01-05  1525162.0     14.237612
        2016-01-06  1424264.0     14.169166
        ...               ...           ...
        2018-10-14  2185230.0     14.597232
        2018-10-15  1825539.0     14.417386
        2018-10-16  1776778.0     14.390313
        2018-10-17  1792899.0     14.399345
        2018-10-18  1738657.0     14.368624
        [1021 rows x 2 columns], ['2016-12-26 00:00:00', '2018-09-10 00:00:00'])
        """
        import numpy as np
        import pandas as pd
        from changepy import pelt
        from changepy.costs import normal_var

        change_point_threshold = self.change_point_threshold

        df_copy = pd.DataFrame(df[metric])

        counts = df_copy[metric].values
        mean = np.mean(counts)

        # Performing changepoint detection with respect to the data variablity shift through PELT
        cdate = pelt(normal_var(counts, mean), len(counts))

        # If PELT detects the first datapoint to be a change point, then we ignore that change point
        if cdate:
            if cdate[0] == 0:
                cdate.remove(0)
        if len(cdate) > 0:
            # Finding the magnitude of divergence around every change point (detected by PELT) by comparing the
            # distributions of the data points on the left and the right side of the change point
            entrp = self._shift_intensity(change_points=cdate, df=df_copy, metric=metric)
            df_change_points = pd.DataFrame({'c_point': cdate, 'entropy': entrp})

            # Narrowing down to the change points which satisfies a required lower bound of divergence
            df_change_points = df_change_points[df_change_points['entropy'] > change_point_threshold]
            cdate = df_change_points['c_point'].values

            # Set the start date of the time series based on the min_ts_length and the max_ts_length and the change
            # points
            if len(cdate) > 0:
                df_subset = df_copy.iloc[cdate]
                change_point_list = [i.__str__() for i in df_subset.index]
                index = df_subset.index[-1]
                copy_df = df.loc[index: df.last_valid_index()] if self.data_shift_truncate else df
                if copy_df.shape[0] < min_ts_length:
                    # Return None, If time series after the change point contains less number of data points
                    # than the minimum required length
                    return None, change_point_list
                elif copy_df.shape[0] < max_ts_length:
                    # Return the time series after the change point if it's length lies between the minimum and the
                    # maximum required length
                    pass
                elif copy_df.shape[0] > max_ts_length:
                    # Truncate the time series after change point if it contains more than required data for
                    # training
                    copy_df = df.iloc[-max_ts_length:]
            else:
                change_point_list = None
                if df.shape[0] < min_ts_length:
                    return None, change_point_list
                else:
                    if df.shape[0] < max_ts_length:
                        copy_df = df
                    else:
                        copy_df = df.iloc[-max_ts_length:]
        else:
            change_point_list = None
            if df.shape[0] < min_ts_length:
                return None, change_point_list
            else:
                if df.shape[0] < max_ts_length:
                    copy_df = df
                else:
                    copy_df = df.iloc[-max_ts_length:]
        return copy_df, change_point_list

    def _trend_changes(self, input_df=None, value_column=None):
        """
        This function detects the trend changes of the input time series
        :param pandas.DataFrame input_df: The input sequence of data.
        :param str value_column: A string containing the column name containing the target series
        :param sig_level: The significance level to identify relevant trend changes.
        :param window_size: The size of the sub windows for input data segmentation.
        :return: list of strings with the potential trend changes.
        :rtype: list[str]

        >>> input_df
                          raw  interpolated
        2016-01-02  1753421.0     14.377080
        2016-01-03  1879108.0     14.446308
        2016-01-04  1462725.0     14.195812
        2016-01-05  1525162.0     14.237612
        2016-01-06  1424264.0     14.169166
        ...               ...           ...
        2018-10-14  2185230.0     14.597232
        2018-10-15  1825539.0     14.417386
        2018-10-16  1776778.0     14.390313
        2018-10-17  1792899.0     14.399345
        2018-10-18  1738657.0     14.368624

        >>> global_trend_changes
        ['2016-04-16 00:00:00', '2016-05-28 00:00:00', '2016-08-06 00:00:00', '2016-11-05 00:00:00',
        '2016-12-31 00:00:00', '2017-02-25 00:00:00', '2017-03-25 00:00:00', '2017-06-03 00:00:00',
        '2017-07-01 00:00:00', '2017-08-26 00:00:00', '2017-09-30 00:00:00', '2017-10-28 00:00:00',
        '2017-12-23 00:00:00', '2018-02-17 00:00:00', '2018-03-17 00:00:00', '2018-05-19 00:00:00',
        '2018-06-30 00:00:00', '2018-09-08 00:00:00']

        """
        import numpy as np
        from scipy import stats
        from statsmodels.tsa.stattools import acf

        min_float = 1e-10
        window_size = self.window_size
        sig_level = self.sig_level

        series = input_df[value_column].tolist()
        timestamps = input_df.index.tolist()

        if not window_size:
            window_size = self._detect_window_size(series=series)

        # Creating a crude estimation of the required window size
        nwindows = int(window_size * 1.5)

        current_mid_point = window_size * nwindows
        past_trend_change = 0
        global_trend_changes = []
        local_trend_changes = {}
        past_p_value = -1

        # If the remaining part of the time series is less than the (window_size * nwindows) we terminate the while loop
        current_reminder = len(series) if (current_mid_point + (window_size * nwindows)) < len(series) else 0
        while current_reminder >= window_size:

            # Creating the left and the right window for slope detection
            l_window = series[current_mid_point - (window_size * nwindows): current_mid_point]
            r_window = series[current_mid_point: current_mid_point + (window_size * nwindows)]
            l_window_size = len(l_window)
            r_window_size = len(r_window)
            N = l_window_size + r_window_size

            # Finding the effective degrees of freedom
            auto_corr = acf(l_window + r_window, nlags=N)
            auto_corr[np.isnan(auto_corr)] = 1
            eff_df = 0
            for i in range(1, N):
                eff_df = eff_df + (((N - i) / float(N)) * auto_corr[i])
            eff_df = max(1, int(np.rint(1 / ((1 / float(N)) + ((2 / float(N)) * eff_df)))) - 4)

            # Creating the left and right indices for running the regression
            l_x, r_x = np.arange(l_window_size), np.arange(r_window_size)

            # Linear regression on the left and the right window
            l_slope, l_intercept, l_r_value, l_p_value, l_std_err = stats.linregress(l_x, l_window)
            r_slope, r_intercept, r_r_value, r_p_value, r_std_err = stats.linregress(r_x, r_window)

            # t-test for slope shift
            l_window_hat = (l_slope * l_x) + l_intercept
            r_window_hat = (r_slope * r_x) + r_intercept

            l_sse = np.sum((l_window - l_window_hat) ** 2)
            r_sse = np.sum((r_window - r_window_hat) ** 2)

            l_const = np.sum((np.arange(1, l_window_size + 1) - ((l_window_size + 1) / 2.0)) ** 2)
            r_const = np.sum((np.arange(1, r_window_size + 1) - ((r_window_size + 1) / 2.0)) ** 2)

            prop_const = (l_const * r_const) / (l_const + r_const)

            total_sse = l_sse + r_sse

            std_err = max(np.sqrt(total_sse / (prop_const * (l_window_size + r_window_size - 4))), min_float)

            t_stat = abs(l_slope - r_slope) / std_err

            p_value = (1 - stats.t.cdf(t_stat, df=eff_df)) * 2

            if p_value < sig_level:
                # Check if the same shift detected multiple times
                if p_value == past_p_value:
                    local_trend_changes.pop(past_trend_change)
                if current_reminder - window_size < window_size:
                    if len(local_trend_changes) > 2:
                        # _local_minima function is called to detec the optimal trend change(s) among a group of local
                        # trend changes
                        current_trend_change = self._local_minima(input_dict=local_trend_changes,
                                                                  window_size=window_size)
                        for key in current_trend_change:
                            global_trend_changes.append(timestamps[key].__str__())
                else:
                    # Handling the trend changes at the tail of the time series
                    local_trend_changes[current_mid_point] = p_value
                    past_trend_change = current_mid_point
                past_p_value = p_value
            else:
                # Handling the trend changes at the tail of the time series
                if (current_mid_point - past_trend_change) <= window_size and len(local_trend_changes) > 2:
                    current_trend_change = self._local_minima(input_dict=local_trend_changes, window_size=window_size)
                    for key in current_trend_change:
                        global_trend_changes.append(timestamps[key].__str__())
                local_trend_changes = {}

            current_mid_point = current_mid_point + window_size
            current_reminder = len(series) - current_mid_point

        return global_trend_changes

    def kf_naive_outlier_detection(self, input_series, idx_position):
        """
        This function detects outlier for the specified index position of the series.

        :param numpy.array input_series: Input time series
        :param int idx_position: Target index position
        :return: Anomaly flag
        :rtype: bool

        >>> input_series = [110, 119, 316, 248, 451, 324, 241, 275, 381]
        >>> self.kf_naive_outlier_detection(input_series, 6)
        False
        """
        import numpy as np
        from pykalman import KalmanFilter

        kf = KalmanFilter()
        filtered_state_means, filtered_state_covariance = kf.em(input_series).filter(input_series)

        residuals = input_series - filtered_state_means[:,0]

        is_anomaly = residuals[idx_position] < np.mean(residuals) \
                     - (2.5 * np.sqrt(filtered_state_covariance)[idx_position][0][0]) \
                     or residuals[idx_position] > np.mean(residuals) \
                     + (2.5 * np.sqrt(filtered_state_covariance)[idx_position][0][0])

        return is_anomaly

    def _truncate_by_data_gaps(self, df, target_metric):
        """
        This function truncates time series after large data gaps.

        :param pandas.DataFrame df: Input time series in pandas data frame
        :param str target_metric: Target value column in the input data frame
        :return: Pandas dataframe with truncated pandas data frame
        :rtype: pandas.DataFrame
        """

        import numpy as np

        max_data_gap = abs(self.min_ts_length / 3.0)

        gap_len = 0
        last_avl_idx = None
        for row in df[::-1].iterrows():
            if np.isnan(row[1][target_metric]) or row[1][target_metric] is None:
                gap_len = gap_len + 1
            else:
                gap_len = 0
                last_avl_idx = row[0]

            if gap_len >= max_data_gap and last_avl_idx:
                truncated_df = df[last_avl_idx:]
                return truncated_df

        return df


    def profile(self, df, impute_only=False, **kwargs):
        """
        This function performs required data profiling and pre-processing before hyperparameter optimization or time
        series model training.

        :param list/pandas.DataFrame df: Input time series.
        :param impute_only: Flag to perform preprocessing until imputation OR full preprocessing.
        :type impute_only: bool, optional
        :return: Preprocessed dataframe with batch data summary.
        :rtype: tuple[pandas.dataFrame, dict]

        >>> de_obj = DataExploration(freq='D', data_shift_truncate=1, is_log_transformed=0, fill_rate=0.9)
        >>> data
                       raw
        index
        2020-01-01  1326.0
        2020-01-02  1552.0
        2020-01-03  1432.0
        2020-01-04  1470.0
        2020-01-05  1565.0
        ...            ...
        2020-06-03  1934.0
        2020-06-04  1873.0
        2020-06-05  1674.0
        2020-06-06  1747.0
        2020-06-07  1782.0
        >>> data, summary = de_obj.profile(data)
        >>> data, summary
        (              raw interpolated
        2020-03-16  1371.0       1371.0
        2020-03-17  1325.0       1325.0
        2020-03-18  1318.0       1318.0
        2020-03-19  1270.0       1270.0
        2020-03-20  1116.0       1116.0
        ...            ...          ...
        2020-06-03  1934.0       1934.0
        2020-06-04  1873.0       1873.0
        2020-06-05  1674.0       1674.0
        2020-06-06  1747.0       1747.0
        2020-06-07  1782.0       1782.0
        [84 rows x 2 columns], {'success': True, 'trend_change_list': ['2020-04-01 00:00:00'], 'change_point_list':
        ['2020-03-16 00:00:00'], 'is_log_transformed': 0, 'min_ts_mean': None, 'ts_start': '2020-01-01 00:00:00',
        'ts_end': '2020-06-07 00:00:00'})

        """

        import numpy as np

        min_ts_length = self.min_ts_length
        max_ts_length = self.max_ts_length
        is_log_transformed = self.is_log_transformed
        target_metric = 'raw'
        imputed_metric = 'interpolated'

        try:
            # if input dimension/metric combination timeseries is null, skip modeling
            if len(df) == 0:
                raise ValueError("No model ran because dimension/metric combination is null")
            if len(df) < min_ts_length:
                raise ValueError("Current time series length of {}{} is less tha minimum requires "
                                 "length {}{} for training".format(len(df), self.freq, min_ts_length, self.freq))

            if isinstance(df, list):
                import pandas as pd
                df = (pd.DataFrame(df, columns=[self._target_index, self._target_metric]).set_index(self._target_index))

            df = self.add_missing_index(df=df, freq=self.freq)

            df = df.iloc[-min(max_ts_length, len(df)):]

            df = self._truncate_by_data_gaps(df=df, target_metric=target_metric)

            if len(df) < min_ts_length:
                raise ValueError("The training data observed continuous missing data near the end. Require more stable "
                                 "data to train")

            df = self._kalman_smoothing_imputation(df=df, target_metric=target_metric, imputed_metric=imputed_metric,
                                                   impute_only=impute_only)

            if impute_only:
                summary = {'success': True}
                return df, summary

            if self.freq == 'D':
                train_end_anomaly_flag = self.kf_naive_outlier_detection(input_series=df['interpolated'].values,
                                                                         idx_position=len(df['interpolated']) - 1)
                if train_end_anomaly_flag:
                    df = df.iloc[:-1]

            if is_log_transformed and not df[df[target_metric] < 0].empty:
                is_log_transformed = False

            # We want to make sure the time series does not contain any negatives in case of log transformation
            if is_log_transformed:
                min_ts_mean = np.log(max(self.min_ts_mean, 1)) if self.min_ts_mean is not None else None
            else:
                min_ts_mean = self.min_ts_mean

            if np.sum(np.isnan(df[target_metric].values[-(2 * min_ts_length):])) / float(2 * min_ts_length) \
                    > 1 - self.fill_rate:
                if np.sum(np.isnan(df[target_metric].values[-min_ts_length:])) > 0:
                    raise ValueError('Too few observed data near the prediction date')

            if 'ts_start' not in kwargs:
                ts_start = df.index.min()
            else:
                ts_start = max(df.index.min(), kwargs['ts_start'])

            if 'ts_end' not in kwargs:
                ts_end = df.index.max()
            else:
                ts_end = min(df.index.max(), kwargs['ts_end'])

            df = df[ts_start:]

            df = df.loc[:ts_end]

            if is_log_transformed:
                df[imputed_metric] = np.log(df[imputed_metric] + 1) if is_log_transformed else df[imputed_metric]

            data, change_point_list = self._pelt_change_point_detection(df=df, metric=imputed_metric,
                                                                        min_ts_length=min_ts_length,
                                                                        max_ts_length=max_ts_length)

            trend_change_list = self._trend_changes(input_df=df, value_column=imputed_metric)

            summary = {'success': True,
                       'trend_change_list': trend_change_list if len(trend_change_list) > 0 else None,
                       'change_point_list': change_point_list,
                       'is_log_transformed': is_log_transformed,
                       'min_ts_mean': min_ts_mean,
                       'ts_start': str(ts_start),
                       'ts_end': str(ts_end)}

        except Exception as e:
            # Data exploration failed
            summary = {'success': False,
                       'ErrorMessage': str(e)}
            data = None

        return data, summary
