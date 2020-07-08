import os
from os.path import dirname as up
import pytest
import pandas as pd
import numpy as np

def get_data_path(path):
    luminaire_test_dir = up(os.path.realpath(path))
    return 'file://' + os.path.join(luminaire_test_dir,
                                    'luminaire',
                                    'tests',
                                    'datasets',
                                    path)

def get_model_path(path):
    luminaire_test_dir = up(os.path.realpath(path))
    return os.path.join(luminaire_test_dir,
                                    'luminaire',
                                    'tests',
                                    'models',
                                    path)

@pytest.fixture(scope='session')
def test_data_with_missing():
    """
    Data with missing indexes to test data exploration
    """

    data = pd.read_csv(get_data_path('daily_test_time_series_with_missing.csv'))
    data['index'] = pd.DatetimeIndex(data['index'])
    data = pd.DataFrame(data, columns=['index', 'raw']).set_index('index')

    return data

@pytest.fixture(scope='session')
def change_test_data():
    """
    Data without missing indexes to test data exploration
    """

    data = pd.read_csv(get_data_path('daily_test_time_series.csv'))
    data['index'] = pd.DatetimeIndex(data['index'])
    data = pd.DataFrame(data, columns=['index', 'raw']).set_index('index')

    return data

@pytest.fixture(scope='session')
def exploration_test_array():
    """
    Data without missing indexes to test data exploration
    """

    data = pd.read_csv(get_data_path('daily_test_time_series.csv'))
    data['index'] = pd.DatetimeIndex(data['index'])
    data = pd.DataFrame(data, columns=['index', 'raw']).set_index('index')['raw'].values

    return data

@pytest.fixture(scope='session')
def training_test_data():
    """
    Data with missing indexes to test lad structural training
    """

    data = pd.read_csv(get_data_path('daily_test_time_series.csv'))
    data['index'] = pd.DatetimeIndex(data['index'])
    data['interpolated'] = data['raw']
    data = pd.DataFrame(data, columns=['index', 'raw', 'interpolated']).set_index('index')

    return data

@pytest.fixture(scope='session')
def scoring_test_data():
    """
    Data with missing indexes to test lad structural training
    """

    data = pd.read_csv(get_data_path('daily_test_time_series_scoring.csv'))
    data['index'] = pd.DatetimeIndex(data['index'])
    data['interpolated'] = data['raw']
    data = pd.DataFrame(data, columns=['index', 'raw', 'interpolated']).set_index('index')

    return data

@pytest.fixture(scope='session')
def training_test_data_log():
    """
    Data with missing indexes to test lad structural training
    """

    data = pd.read_csv(get_data_path('daily_test_time_series_seasonal.csv'))
    data['index'] = pd.DatetimeIndex(data['index'])
    data['interpolated'] = np.log(data['raw'] + 1)
    data = pd.DataFrame(data, columns=['index', 'raw', 'interpolated']).set_index('index')

    return data

@pytest.fixture(scope='session')
def scoring_test_data_log():
    """
    Data with missing indexes to test lad structural training
    """

    data = pd.read_csv(get_data_path('daily_test_time_series_seasonal_scoring.csv'))
    data['index'] = pd.DatetimeIndex(data['index'])
    data['interpolated'] = np.log(data['raw'] + 1)
    data = pd.DataFrame(data, columns=['index', 'raw', 'interpolated']).set_index('index')

    return data

@pytest.fixture(scope='session')
def lad_structural_model():
    """
    Model to test lad structural result
    """
    import pickle

    model_file = open(get_model_path('lad_structural_model'), 'rb')
    model = pickle.load(model_file)

    model_file.close()

    return model

@pytest.fixture(scope='session')
def lad_filtering_model():
    """
    Model to test lad filtering model result
    """
    import pickle

    model_file = open(get_model_path('lad_filtering_model'), 'rb')
    model = pickle.load(model_file)

    model_file.close()

    return model

@pytest.fixture(scope='session')
def lad_structural_model_log_seasonal():
    """
    Model to test lad structural result
    """
    import pickle

    model_file = open(get_model_path('lad_structural_model_log_seasonal'), 'rb')
    model = pickle.load(model_file)

    model_file.close()

    return model

@pytest.fixture(scope='session')
def lad_filtering_model_log_seasonal():
    """
    Model to test lad filtering model result
    """
    import pickle

    model_file = open(get_model_path('lad_filtering_model_log_seasonal'), 'rb')
    model = pickle.load(model_file)

    model_file.close()

    return model

@pytest.fixture(scope='session')
def window_density_model_data():
    """
    Window based training for high frequency data
    """

    data = pd.read_csv(get_data_path('window_density_test_10_mins.csv'))
    data['interpolated'] = data['raw']
    data['index'] = pd.DatetimeIndex(data['index'])
    data = data.set_index('index')

    return data

@pytest.fixture(scope='session')
def window_density_model():
    """
    Model to test Window Density model result
    """
    import pickle

    model_file = open(get_model_path('window_density_model'), 'rb')
    model = pickle.load(model_file)

    model_file.close()

    return model

@pytest.fixture(scope='session')
def window_density_model_data_hourly():
    """
    Window based training for high frequency data
    """

    data = pd.read_csv(get_data_path('window_density_test_hourly.csv'))
    data['interpolated'] = data['raw']
    data['index'] = pd.DatetimeIndex(data['index'])
    data = data.set_index('index')

    return data

@pytest.fixture(scope='session')
def window_density_model_hourly_last_window():
    """
    Model to test Window Density model result
    """
    import pickle

    model_file = open(get_model_path('window_density_model_hourly_last_window'), 'rb')
    model = pickle.load(model_file)

    model_file.close()

    return model

@pytest.fixture(scope='session')
def window_density_model_hourly_aggregated():
    """
    Model to test Window Density model result
    """
    import pickle

    model_file = open(get_model_path('window_density_model_hourly_aggregated'), 'rb')
    model = pickle.load(model_file)

    model_file.close()

    return model