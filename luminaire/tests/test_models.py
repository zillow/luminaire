from luminaire.model.lad_structural import *
from luminaire.model.lad_filtering import *
from luminaire.model.window_density import *
from datetime import datetime

class TestLADStructural(object):

    def test_lad_structural_training(self, training_test_data):

        hyper_params = LADStructuralHyperParams(is_log_transformed=False, p=4, q=0).params
        lad_struct_obj = LADStructuralModel(hyper_params, freq='D')
        data_summary = {'ts_start': training_test_data.first_valid_index(),
                        'ts_end': training_test_data.last_valid_index(),
                        'is_log_transformed': False}
        success, ts_end, model = lad_struct_obj.train(data=training_test_data, **data_summary)

        assert success and isinstance(model, LADStructuralModel)

    def test_lad_structural_training_zeroes(self, training_test_data_zeroes):

        hyper_params = LADStructuralHyperParams(is_log_transformed=False, p=4, q=0).params
        lad_struct_obj = LADStructuralModel(hyper_params, freq='D')
        data_summary = {'ts_start': training_test_data_zeroes.first_valid_index(),
                        'ts_end': training_test_data_zeroes.last_valid_index(),
                        'is_log_transformed': False}
        success, ts_end, model = lad_struct_obj.train(data=training_test_data_zeroes, **data_summary)

        assert success and isinstance(model, LADStructuralModel)


    def test_lad_structural_scoring(self, scoring_test_data, lad_structural_model):

        pred_date_normal = scoring_test_data.index[0]
        value_normal = scoring_test_data['raw'].iloc[0]
        output_normal = lad_structural_model.score(value_normal, pred_date_normal)

        pred_date_anomalous = scoring_test_data.index[1]
        value_anomalous = scoring_test_data['raw'].iloc[1]
        output_anomalous = lad_structural_model.score(value_anomalous, pred_date_anomalous)

        assert output_normal['Success'] and not output_normal['IsAnomaly']
        assert output_anomalous['Success'] and output_anomalous['IsAnomaly']

    def test_lad_filtering_training(self, training_test_data):

        hyper_params = LADFilteringHyperParams(is_log_transformed=False).params
        lad_filtering_obj = LADFilteringModel(hyper_params, freq='D')
        data_summary = {'ts_start': training_test_data.first_valid_index(),
                        'ts_end': training_test_data.last_valid_index(),
                        'is_log_transformed': False}
        success, ts_end, model = lad_filtering_obj.train(data=training_test_data, **data_summary)

        assert success and isinstance(model, LADFilteringModel)

    def test_lad_filtering_scoring(self, scoring_test_data, lad_filtering_model):

        pred_date_normal = scoring_test_data.index[0]
        value_normal = scoring_test_data['raw'].iloc[0]
        output_normal, lad_filtering_model_update = lad_filtering_model.score(value_normal, pred_date_normal)

        pred_date_anomalous = scoring_test_data.index[1]
        value_anomalous = scoring_test_data['raw'].iloc[1]
        output_anomalous, lad_filtering_model_update = lad_filtering_model_update.score(value_anomalous, pred_date_anomalous)

        assert output_normal['Success'] and not output_normal['IsAnomaly']
        assert output_anomalous['Success'] and output_anomalous['IsAnomaly'] \
               and isinstance(lad_filtering_model_update, LADFilteringModel)

    def test_lad_structural_training_log(self, training_test_data_log):

        hyper_params = LADStructuralHyperParams(is_log_transformed=True, include_holidays_exog=False).params
        lad_structural_obj = LADStructuralModel(hyper_params, freq='D')
        data_summary = {'ts_start': training_test_data_log.first_valid_index(),
                        'ts_end': training_test_data_log.last_valid_index(),
                        'is_log_transformed': True}
        success, ts_end, model = lad_structural_obj.train(data=training_test_data_log, **data_summary)

        assert success and isinstance(model, LADStructuralModel)

    def test_lad_structural_scoring_log(self, scoring_test_data_log, lad_structural_model_log_seasonal):

        pred_date_normal = scoring_test_data_log.index[0]
        value_normal = scoring_test_data_log['raw'].iloc[0]
        output_normal = lad_structural_model_log_seasonal.score(value_normal, pred_date_normal)

        pred_date_anomalous = scoring_test_data_log.index[1]
        value_anomalous = scoring_test_data_log['raw'].iloc[1]
        output_anomalous = lad_structural_model_log_seasonal.score(value_anomalous, pred_date_anomalous)

        assert output_normal['Success'] and output_normal['IsAnomaly']
        assert output_anomalous['Success'] and output_anomalous['IsAnomaly']

    def test_lad_filtering_training_log(self, training_test_data_log):

        hyper_params = LADFilteringHyperParams(is_log_transformed=True).params
        lad_filtering_obj = LADFilteringModel(hyper_params, freq='D')
        data_summary = {'ts_start': training_test_data_log.first_valid_index(),
                        'ts_end': training_test_data_log.last_valid_index(),
                        'is_log_transformed': True}
        success, ts_end, model = lad_filtering_obj.train(data=training_test_data_log, **data_summary)

        assert success and isinstance(model, LADFilteringModel)

    def test_lad_filtering_scoring_log(self, scoring_test_data_log, lad_filtering_model_log_seasonal):

        pred_date_normal = scoring_test_data_log.index[0]
        value_normal = scoring_test_data_log['raw'].iloc[0]
        output_normal, lad_filtering_model_update = lad_filtering_model_log_seasonal.score(value_normal, pred_date_normal)

        pred_date_anomalous = scoring_test_data_log.index[1]
        value_anomalous = scoring_test_data_log['raw'].iloc[1]
        output_anomalous, lad_filtering_model_update = lad_filtering_model_update.score(value_anomalous, pred_date_anomalous)

        assert output_normal['Success'] and not output_normal['IsAnomaly']
        assert output_anomalous['Success'] and output_anomalous['IsAnomaly'] \
               and isinstance(lad_filtering_model_update, LADFilteringModel)

    def test_high_freq_window_density_training(self, window_density_model_data):
        training_start = datetime(2020, 4, 30)
        training_end = datetime(2020, 5, 27, 23, 59, 59)
        data = window_density_model_data[(window_density_model_data.index >= training_start)
                                                              & (window_density_model_data.index <= training_end)]

        config = WindowDensityHyperParams(detection_method='kldiv',
                                          window_length=6 * 24).params
        de_obj = DataExploration(**config)
        data, pre_prc = de_obj.stream_profile(df=data)
        config.update(pre_prc)
        wdm_obj = WindowDensityModel(hyper_params=config)
        success, ts_end, model = wdm_obj.train(data=data, past_model=None)

        assert success and isinstance(model, WindowDensityModel)

    def test_high_freq_window_density_scoring(self, window_density_model_data, window_density_model):
        scoring_start = datetime(2020, 5, 28)
        scoring_end = datetime(2020, 5, 28, 23, 59, 59)

        data = window_density_model_data[(window_density_model_data.index >= scoring_start)
                                         & (window_density_model_data.index <= scoring_end)]

        result = window_density_model.score(data)

        assert result[0]['Success'] and isinstance(result[0]['AnomalyProbability'], float)

    def test_low_freq_window_density_training_last_window(self, window_density_model_data_hourly):
        training_start = datetime(2018, 4, 1)
        training_end = datetime(2018, 9, 30, 23, 59, 59)
        data = window_density_model_data_hourly[(window_density_model_data_hourly.index >= training_start)
                                                & (window_density_model_data_hourly.index <= training_end)]
        config = WindowDensityHyperParams(freq='H', baseline_type="last_window").params
        de_obj = DataExploration(**config)
        data, pre_prc = de_obj.stream_profile(df=data)
        config.update(pre_prc)
        wdm_obj = WindowDensityModel(hyper_params=config)
        success, ts_end, model = wdm_obj.train(data=data, past_model=None)

        assert success and isinstance(model, WindowDensityModel)

    def test_low_freq_window_density_scoring_last_window(self, window_density_model_data_hourly,
                                                         window_density_model_hourly_last_window):
        scoring_start = datetime(2018, 10, 1)
        scoring_end = datetime(2018, 10, 1, 23, 59, 59)

        data = window_density_model_data_hourly[(window_density_model_data_hourly.index >= scoring_start)
                                                & (window_density_model_data_hourly.index <= scoring_end)]

        result = window_density_model_hourly_last_window.score(data)

        assert result[0]['Success'] and isinstance(result[0]['AnomalyProbability'], float)

    def test_low_freq_window_density_training_aggregated(self, window_density_model_data_hourly):
        training_start = datetime(2018, 4, 1)
        training_end = datetime(2018, 9, 30, 23, 59, 59)
        data = window_density_model_data_hourly[(window_density_model_data_hourly.index >= training_start)
                                                & (window_density_model_data_hourly.index <= training_end)]
        config = WindowDensityHyperParams(freq='H', baseline_type="aggregated").params
        de_obj = DataExploration(**config)
        data, pre_prc = de_obj.stream_profile(df=data)
        config.update(pre_prc)
        wdm_obj = WindowDensityModel(hyper_params=config)
        success, ts_end, model = wdm_obj.train(data=data, past_model=None)

        assert success and isinstance(model, WindowDensityModel)

    def test_low_freq_window_density_scoring_aggregated(self, window_density_model_data_hourly,
                                                        window_density_model_hourly_aggregated):
        scoring_start = datetime(2018, 10, 1)
        scoring_end = datetime(2018, 10, 1, 23, 59, 59)

        data = window_density_model_data_hourly[(window_density_model_data_hourly.index >= scoring_start)
                                                & (window_density_model_data_hourly.index <= scoring_end)]

        result = window_density_model_hourly_aggregated.score(data)

        assert result[0]['Success'] and isinstance(result[0]['AnomalyProbability'], float)

    def test_lad_filtering_scoring_diff_order(self, scoring_test_data, lad_filtering_model):
        import numpy as np
        # check to see if scoring yields AdjustedActual with correct order of differences
        pred_date_normal = scoring_test_data.index[0]
        value_normal = scoring_test_data['raw'].iloc[0]
        output_normal, lad_filtering_model_update = lad_filtering_model.score(value_normal, pred_date_normal)
        # collect data
        diff_order = output_normal["NonStationarityDiffOrder"]
        adj_actual  = output_normal["AdjustedActual"]
        last_points = lad_filtering_model._params['last_data_points']
        last_points.append(value_normal)
        # diff with model's diff_order
        diff = np.diff(last_points, diff_order)[-1]

        assert diff == adj_actual, f"AdjustedActual {adj_actual} does not match diff {diff_order} of last_data_points {last_points}"
