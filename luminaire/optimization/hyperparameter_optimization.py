from hyperopt import fmin, tpe, hp, STATUS_OK
from luminaire.model import LADStructuralModel, LADStructuralHyperParams, LADFilteringModel, LADFilteringHyperParams
from luminaire.exploration.data_exploration import DataExploration
import warnings
warnings.filterwarnings('ignore')


class HyperparameterOptimization(object):
    """
    Hyperparameter optimization for LAD outlier detection configuration for batch data.

    :param str freq: The frequency of the time-series. A `Pandas offset`_ such as 'D', 'H', or 'M'.
    :param str detection_type: Luminaire anomaly detection type. Only Outlier detection for batch data is currently
        supported.
    :type detection_type: str, optional
    :param min_ts_mean: Minimum average values in the most recent window of the time series. This optional parameter
        can be used to avoid over-alerting from noisy low volume time series.
    :type min_ts_mean: float, optional
    :param max_ts_length: The maximum required length of the time series for training.
    :type max_ts_length: int, optional
    :param min_ts_length: The minimum required length of the time series for training.
    :type min_ts_length: int, optional
    :param int scoring_length: Number of innovations to be scored after training window with respect to the frequency.
    :type scoring_length: int, optional

    .. _Pandas offset: https://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases
    """

    def __init__(self,
                 freq,
                 detection_type='OutlierDetection',
                 min_ts_mean=None,
                 max_ts_length=None,
                 min_ts_length=None,
                 scoring_length=None,
                 **kwargs):
        self._target_metric = 'raw'
        self.freq = freq
        self.detection_type = detection_type
        self.min_ts_mean=min_ts_mean
        self._target_index = 'index'
        self._target_metric = 'raw'
        self.max_ts_length = max_ts_length
        self.min_ts_length = min_ts_length
        self.anomaly_intensity_list = [-0.6, -0.3, -0.1, 0.1, 0.3, 0.6]

        scoring_length_dict = {
            'H': 36, 'D': 10, 'W': 8, 'M': 6,
        }
        self.scoring_length = scoring_length or (scoring_length_dict.get(freq)
                                                 if freq in scoring_length_dict.keys() else 30)

    def _mdape(self, actuals, predictions):
        """
        This function computes the median absolute percentage error for the observed vs the predicted values.

        :param numpy.array actuals: Observed values
        :param numpy.array predictions: Predicted values
        :return: Mean absolute percentage error

        :rtype: numpy.mean
        """
        import numpy as np

        actuals = np.array(actuals)
        predictions = np.array(predictions)

        non_zero_idx = set(np.argwhere(actuals)[:,0].tolist())
        non_none_idx = set(np.where(~np.isnan(predictions.astype(float)))[0].tolist())
        filtered_idx = list(non_zero_idx.intersection(non_none_idx))
        filtered_actuals = actuals[filtered_idx]
        filtered_predictions = predictions[filtered_idx]

        mdape = np.median(np.abs((filtered_actuals - filtered_predictions) / filtered_actuals))

        return mdape if not np.isnan(mdape) else None

    def _synthetic_anomaly_check(self, observation, prediction, std_error):
        """
        This function performs anomaly detection based on synthetic anomalies for a given anomaly intensity list

        :param float observation: Observed value
        :param float prediction: Predicted value
        :param float std_error: Standard error for the predictive model
        :return: return the list of anomaly flags with the corresponding probabilities
        :rtype: tuple[list, list]
        """

        import numpy as np
        import scipy.stats as st
        float_min = 1e-10

        anomaly_flags = []
        anomaly_probabilities = []

        # Anomaly detection based on synthetic anomalies generated through a given intensity list
        for prop in self.anomaly_intensity_list:
            trial_prob = np.random.uniform(0, 1, 1)
            if trial_prob < 0.4:
                synthetic_value = observation + (prop * observation)
                anomaly_flags.append(1)
            else:
                synthetic_value = observation
                anomaly_flags.append(0)

            zscore_abs = abs((synthetic_value - prediction) / max(float(std_error), float_min))
            probability = (2 * st.norm(0, 1).cdf(zscore_abs)) - 1
            anomaly_probabilities.append(probability[0])

        return anomaly_flags, anomaly_probabilities

    def _objective_part(self, data, smoothed_series, args):
        """
        This is the objective function that outputs the loss for a giveen set of hyperparameters for optimization
        through hyperopt

        :param pandas.DataFrame data: Input time series data
        :param list smoothed_series: Input time series after smoothing
        :param args:
        :return: AUC based on observed (synthetic) and predicted anomalies
        :rtype: dict

        >>> data
                          raw
        2016-01-02  1753421.0
        2016-01-03  1879108.0
        2016-01-04  1462725.0
        2016-01-05  1525162.0
        2016-01-06  1424264.0
        ...               ...
        2018-10-24  1726884.0
        2018-10-25  1685651.0
        2018-10-26  1632952.0
        2018-10-27  1850912.0
        2018-10-28  2021929.0

        >>> {'loss': 1 - auc, 'status': STATUS_OK}
        {'loss': 0.3917824074074072, 'status': 'ok'}
        """

        import numpy as np
        import pandas as pd
        from sklearn.metrics import log_loss
        import copy

        is_log_transformed = args[0]
        data_shift_truncate = args[1]
        fill_rate = args[2]

        # Getting hyperparameters for lad structural model
        if args[3]['model'] == 'LADStructuralModel':
            max_ft_freq = args[3]['param']['max_ft_freq']
            include_holidays_exog = args[3]['param']['include_holidays_exog']
            p = args[3]['param']['p']
            q = args[3]['param']['q']

        ts_start = data.index.min()
        ts_end = data.index.max()
        max_ts_length = self.max_ts_length
        min_ts_length = self.min_ts_length
        freq = self.freq
        scoring_length = self.scoring_length

        train_end = (pd.Timestamp(ts_end) - pd.Timedelta("{}".format(scoring_length) + freq)).to_pydatetime()
        score_start = (pd.Timestamp(train_end) + pd.Timedelta("1" + freq)).to_pydatetime()

        training_data = data.loc[ts_start:train_end]
        scoring_data = data.loc[score_start:ts_end]

        try:
            # Required data preprocessing before training and scoring
            de_obj = DataExploration(freq=self.freq, min_ts_length=self.min_ts_length,
                                     min_ts_mean=self.min_ts_mean,
                                     max_ts_length=self.max_ts_length,
                                     is_log_transformed=is_log_transformed,
                                     data_shift_truncate=data_shift_truncate,
                                     detection_type=self.detection_type,
                                     fill_rate=fill_rate)
            training_data, preprocess_summary = de_obj.profile(df=training_data)

            is_log_transformed = preprocess_summary['is_log_transformed']

            # Getting De-noised smoothed series for generating synthetic anomalies
            smoothed_scoring_series = smoothed_series[-len(scoring_data):]

            labels = []
            probs = []

            if args[3]['model'] == 'LADStructuralModel':
                # LAD structural training and scoring
                hyper_params = LADStructuralHyperParams(is_log_transformed=is_log_transformed, max_ft_freq=max_ft_freq,
                                                include_holidays_exog=include_holidays_exog,
                                                p=p, q=q)
                lad_struct = LADStructuralModel(hyper_params.params, max_ts_length=max_ts_length,
                                                min_ts_length=min_ts_length, freq=freq)
                success, model_date, model = lad_struct.train(data=training_data, optimize=True, **preprocess_summary)

                scr_idx = 0

                obs = []
                preds = []
                # Scoring and anomaly classification for synthetic anomalies
                for i, row in scoring_data.iterrows():
                    observed_value = row.raw
                    obs.append(observed_value)
                    result = model.score(observed_value, i)
                    prediction = result['Prediction']
                    preds.append(prediction)
                    std_error = result['StdErr']
                    observation = smoothed_scoring_series[scr_idx]
                    scr_idx = scr_idx + 1
                    anomaly_flags, anomaly_probabilities = self._synthetic_anomaly_check(prediction=prediction,
                                                                                         std_error=std_error,
                                                                                         observation=observation)
                    labels = labels + anomaly_flags
                    probs = probs + anomaly_probabilities

                mdape = self._mdape(obs, preds)
            elif args[3]['model'] == 'LADFilteringModel':
                # LAD filtering training and scoring
                hyper_params = LADFilteringHyperParams(is_log_transformed=is_log_transformed)
                lad_filtering = LADFilteringModel(hyper_params.params, max_ts_length=max_ts_length,
                                                  min_ts_length=min_ts_length, freq=freq)

                success, model_date, stable_model = lad_filtering.train(training_data, **preprocess_summary)
                # Scoring and anomaly classification for synthetic anomalies
                for prop in self.anomaly_intensity_list:
                    anomaly_flags_list = []
                    anomaly_probabilities_list = []
                    local_model = copy.deepcopy(stable_model)
                    for i, row in scoring_data.iterrows():
                        trial_prob = np.random.uniform(0, 1, 1)
                        observed_value = row.raw
                        synthetic_actual = observed_value
                        if trial_prob < 0.4:
                            synthetic_actual = observed_value + (prop * observed_value)
                            anomaly_flags_list.append(1)
                        else:
                            anomaly_flags_list.append(0)

                        result, local_model = local_model.score(observed_value=observed_value, pred_date=i,
                                                                synthetic_actual=synthetic_actual)
                        anomaly_probabilities_list.append(result['AnomalyProbability'])

                    labels = labels + anomaly_flags_list
                    probs = probs + anomaly_probabilities_list

            weights = ((1 - np.array(labels)) + 1) / float(len(labels))
            if args[3]['model'] == 'LADStructuralModel' and mdape:
                cost = (0.5 * mdape) + (0.5 * log_loss(labels, probs, sample_weight=weights))
            else:
                cost = log_loss(labels, probs, sample_weight=weights)

        except Exception as e:
            return {'loss': 1e100, 'status': STATUS_OK}

        return {'loss': cost, 'status': STATUS_OK}

    def _optimize(self, data, objective_part, algo=tpe.suggest, max_evals=50):
        """
        Optimization function that calls the hyperopt for a given set of hyperparameters
        :param pandas.dataFrame data: Input time series data
        :param python function objective_part: Partial objective function with the hyperparameters only
        :param str algo: hyperopt optimization algorithm
        :param int max_evals: Maximum number of evaluation
        :return: Optimal hyperparameters
        :rtype: dict
        """

        from functools import partial
        from pykalman import KalmanFilter

        # detection_type: [OutlierDetection, LaggedOutlierDetection, VolumeDropDetection]
        if self.detection_type == 'OutlierDetection':

            hyper_param_list = [
                {'model': 'LADStructuralModel', 'param': {'max_ft_freq': hp.randint('max_ft_freq', 6) + 2,
                                             'include_holidays_exog': hp.randint('include_holidays_exog', 1) + 1
                                             if self.freq == 'D' else hp.randint('include_holidays_exog', 1),
                                             'p': hp.randint('p', 6) + 1,
                                             'q': hp.randint('q', 6) + 1}
                 },
                {'model': 'LADFilteringModel'}]

            space = [hp.randint('is_log_transformed',2),
                     hp.randint('data_shift_truncate', 2),
                     hp.uniform('fill_rate', 0.7, 1.0),
                     hp.choice('LuminaireModel', [
                     {'model': hyper_param_list[0]['model'], 'param': hyper_param_list[0]['param']},
                     {'model': hyper_param_list[1]['model']}])]

            try:
                series = data[self._target_metric].values
                kf = KalmanFilter()
                smoothed_series, cov_series = kf.em(series).smooth(series)
            except:
                raise ValueError('Kalman Smoothing requires more than one data point')

            objective = partial(objective_part, data, smoothed_series)

        else:
            raise ValueError('Only `detection_type=OutlierDetection` is supported in hyperparameter optimization right now')

        # Calling the optimization function
        hyper_param = fmin(objective, space=space, algo=algo, max_evals=max_evals, show_progressbar=True)
        hyper_param['LuminaireModel'] = hyper_param_list[hyper_param['LuminaireModel']]['model']
        if 'max_ft_freq' in hyper_param:
            hyper_param['max_ft_freq'] = hyper_param['max_ft_freq'] + 2
        if 'include_holidays_exog' in hyper_param and self.freq == 'D':
            hyper_param['include_holidays_exog'] = hyper_param['include_holidays_exog'] + 1
        if 'p' in hyper_param:
            hyper_param['p'] = hyper_param['p'] + 1
        if 'q' in hyper_param:
            hyper_param['q'] = hyper_param['q'] + 1

        return hyper_param

    def run(self, data, max_evals=50):
        """
        This function runs hyperparameter optimization fort LAD batch outlier detection models

        :param list[list] data: Input time series.
        :param int max_evals: Number of iterations for hyperparameter optimization.
        :type max_evals: int, optional
        :return: Optimal hyperparameters.
        :rtype: dict

        >>> data
        [[Timestamp('2020-01-01 00:00:00'), 1326.0],
        [Timestamp('2020-01-02 00:00:00'), 1552.0],
        [Timestamp('2020-01-03 00:00:00'), 1432.0],
        . . . ,
        [Timestamp('2020-06-06 00:00:00'), 1747.0],
        [Timestamp('2020-06-07 00:00:00'), 1782.0]]
        >>> hopt_obj = HyperparameterOptimization(freq='D', detection_type='OutlierDetection')
        >>> hyper_params = hopt_obj._run(data=data, max_evals=5)

        >>> hyper_params
        {'LuminaireModel': 'LADStructuralModel', 'data_shift_truncate': 0, 'fill_rate': 0.8409249603686499,
        'include_holidays_exog': 1, 'is_log_transformed': 1, 'max_ft_freq': 3, 'p': 4, 'q': 3}
        """

        # Calling data exploration to perform imputation only
        de_obj = DataExploration(freq=self.freq, detection_type=self.detection_type)
        data, summary = de_obj.profile(df=data, impute_only=True)

        if summary['success']:
            return self._optimize(data=data, objective_part=self._objective_part, max_evals=max_evals)
        else:
            return None
