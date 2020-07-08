from luminaire.exploration.data_exploration import *
import numpy as np
import pandas as pd

class TestDataExploration(object):

    def test_reindexing_imputation(self, test_data_with_missing):

        de_obj = DataExploration()
        df_after_step1 = de_obj.add_missing_index(df=test_data_with_missing, freq=de_obj.freq)
        df_after_step2 = de_obj._kalman_smoothing_imputation(df=df_after_step1, target_metric='raw', impute_only=True)

        assert len(df_after_step1) == len(pd.date_range(start=test_data_with_missing.first_valid_index(),
                                                        end=test_data_with_missing.last_valid_index(),
                                                        freq=de_obj.freq))
        assert not df_after_step2['raw'].isna().any()

    def test_data_adjustments(self, exploration_test_array):

        from statsmodels.tsa.stattools import adfuller

        de_obj = DataExploration()
        array_after_ma = de_obj._moving_average(exploration_test_array, 14, 28)
        array_after_stat = de_obj._stationarizer(exploration_test_array)
        window_size = de_obj._detect_window_size(exploration_test_array)

        assert len(array_after_ma) == len(exploration_test_array)
        assert adfuller(array_after_stat[0])[1] < 0.05
        assert window_size > 0

    def test_data_change_detection(self, change_test_data):

        de_obj = DataExploration()
        changepoint_output = de_obj._pelt_change_point_detection(change_test_data, 'raw', 21, 3 * 365)
        trendchange_output = de_obj._trend_changes(change_test_data, 'raw')

        assert isinstance(changepoint_output[1], list) and len(changepoint_output[1]) > 0
        assert isinstance(trendchange_output, list) and len(trendchange_output) == 0

    def test_data_profile(self, test_data_with_missing):

        de_obj = DataExploration(fill_rate=0.6, is_log_transformed=False, data_shift_truncate=True)
        data, summary = de_obj.profile(test_data_with_missing)

        assert len(data) > 0 and summary['success']