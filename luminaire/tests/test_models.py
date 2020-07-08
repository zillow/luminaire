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

    def test_lad_structural_scoring(self, scoring_test_data, lad_structural_model):

        pred_date_normal = scoring_test_data.index[0]
        value_normal = scoring_test_data['raw'][0]
        output_normal = lad_structural_model.score(value_normal, pred_date_normal)

        pred_date_anomalous = scoring_test_data.index[1]
        value_anomalous = scoring_test_data['raw'][1]
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
        value_normal = scoring_test_data['raw'][0]
        output_normal, lad_filtering_model_update = lad_filtering_model.score(value_normal, pred_date_normal)

        pred_date_anomalous = scoring_test_data.index[1]
        value_anomalous = scoring_test_data['raw'][1]
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
        value_normal = scoring_test_data_log['raw'][0]
        output_normal = lad_structural_model_log_seasonal.score(value_normal, pred_date_normal)

        pred_date_anomalous = scoring_test_data_log.index[1]
        value_anomalous = scoring_test_data_log['raw'][1]
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
        value_normal = scoring_test_data_log['raw'][0]
        output_normal, lad_filtering_model_update = lad_filtering_model_log_seasonal.score(value_normal, pred_date_normal)

        pred_date_anomalous = scoring_test_data_log.index[1]
        value_anomalous = scoring_test_data_log['raw'][1]
        output_anomalous, lad_filtering_model_update = lad_filtering_model_update.score(value_anomalous, pred_date_anomalous)

        assert output_normal['Success'] and not output_normal['IsAnomaly']
        assert output_anomalous['Success'] and output_anomalous['IsAnomaly'] \
               and isinstance(lad_filtering_model_update, LADFilteringModel)

    def test_high_freq_window_density_training(self, window_density_model_data):
        training_start = datetime(2020, 4, 30)
        training_end = datetime(2020, 5, 25, 23, 59, 59)
        hyper_params = WindowDensityHyperParams(freq='custom', detection_method='kldiv', window_length=6 * 24,
                                                min_window_length=6 * 6, max_window_length=6 * 48,
                                                ma_window_length=24, is_log_transformed=True).params
        wdm_obj = WindowDensityModel(hyper_params=hyper_params)
        success, model = wdm_obj.train(data=window_density_model_data,
                                       training_start=training_start,
                                       training_end=training_end)

        assert success and isinstance(model, WindowDensityModel)

    def test_high_freq_window_density_scoring(self, window_density_model_data, window_density_model):
        scoring_start = datetime(2020, 5, 26)
        scoring_end = datetime(2020, 5, 26, 23, 59, 59)

        data = window_density_model_data[(window_density_model_data.index >= scoring_start)
                                         & (window_density_model_data.index <= scoring_end)]

        result = window_density_model.score(data)

        assert result['Success'] and isinstance(result['AnomalyProbability'], float)

    def test_low_freq_window_density_training_last_window(self, window_density_model_data_hourly):
        training_start = datetime(2018, 4, 1)
        training_end = datetime(2018, 9, 30, 23, 59, 59)
        hyper_params = WindowDensityHyperParams(freq='H', baseline_type="last_window").params
        wdm_obj = WindowDensityModel(hyper_params=hyper_params)
        success, model = wdm_obj.train(data=window_density_model_data_hourly,
                                       training_start=training_start,
                                       training_end=training_end)

        assert success and isinstance(model, WindowDensityModel)

    def test_low_freq_window_density_scoring_last_window(self, window_density_model_data_hourly,
                                                         window_density_model_hourly_last_window):
        scoring_start = datetime(2018, 10, 1)
        scoring_end = datetime(2018, 10, 1, 23, 59, 59)

        data = window_density_model_data_hourly[(window_density_model_data_hourly.index >= scoring_start)
                                                & (window_density_model_data_hourly.index <= scoring_end)]

        result = window_density_model_hourly_last_window.score(data)

        assert result['Success'] and isinstance(result['AnomalyProbability'], float)

    def test_low_freq_window_density_training_aggregated(self, window_density_model_data_hourly):
        training_start = datetime(2018, 4, 1)
        training_end = datetime(2018, 9, 30, 23, 59, 59)
        hyper_params = WindowDensityHyperParams(freq='H').params
        wdm_obj = WindowDensityModel(hyper_params=hyper_params)
        success, model = wdm_obj.train(data=window_density_model_data_hourly,
                                       training_start=training_start,
                                       training_end=training_end)

        assert success and isinstance(model, WindowDensityModel)

    def test_low_freq_window_density_scoring_aggregated(self, window_density_model_data_hourly,
                                                        window_density_model_hourly_aggregated):
        scoring_start = datetime(2018, 10, 1)
        scoring_end = datetime(2018, 10, 1, 23, 59, 59)

        data = window_density_model_data_hourly[(window_density_model_data_hourly.index >= scoring_start)
                                                & (window_density_model_data_hourly.index <= scoring_end)]

        result = window_density_model_hourly_aggregated.score(data)

        assert result['Success'] and isinstance(result['AnomalyProbability'], float)
