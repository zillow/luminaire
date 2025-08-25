import pickle
import pandas as pd
from pathlib import Path
from luminaire.optimization.hyperparameter_optimization import HyperparameterOptimization
from luminaire.model.window_density import WindowDensityHyperParams, WindowDensityModel
from luminaire.exploration.data_exploration import DataExploration
import logging
import json

logging.basicConfig(level=logging.DEBUG)

def get_paths(input_file, output_file):
    input_folder =Path(__file__).parent.joinpath('luminaire', 'tests','datasets')
    output_folder =Path(__file__).parent.joinpath('luminaire', 'tests','models')
    
    input_file = input_folder.joinpath(input_file)
    output_file = output_folder.joinpath(output_file)

    print(f"Input File: {input_file}")
    print(f"Output File: {output_file}")

    return (input_file, output_file)

def reindex(data, format):
    data.index = pd.to_datetime(data['index'], format=format)
    return data

def save_window_density(input_file, output_file):

    input_path, output_path = get_paths(input_file, output_file)
    data = pd.read_csv(input_path)

    # Input data should have a time column set as the index column of the dataframe and a value column named as 'raw'
    data = reindex(data, format='%Y-%m-%d %H:%M:%S')
    import pdb;pdb.set_trace()
    # Optimization
    config = WindowDensityHyperParams().params

    # Profiling
    de_obj = DataExploration(**config)
    training_data, pre_prc = de_obj.stream_profile(df=data)

    if not pre_prc['success']:
        print(f"Window density profiling failed: {pre_prc['ErrorMessage']}")
        print(json.dumps(config, indent=4))
        print(json.dumps(pre_prc, indent=4))
        raise Exception(f"Window density profiling failed: {pre_prc['ErrorMessage']}")



    # Identify Model
    model_class_name = config['LuminaireModel']
    print(f'File: {input_file} Model: {model_class_name}')
    module = __import__('luminaire.model', fromlist=[''])
    model_class = getattr(module, model_class_name)

    # Training
    model_object = model_class(hyper_params=config)
    success, model_date, trained_model = model_object.train(data=training_data, **pre_prc)

    if not success:
        raise Exception(f"Failed to train model {model_class_name} with file: {input_file}")

    with open(output_path, 'wb') as output:
        pickle.dump(trained_model, output)


def save_structural(input_file, output_file):

    input_path, output_path = get_paths(input_file, output_file)
    data = pd.read_csv(input_path)

    # Input data should have a time column set as the index column of the dataframe and a value column named as 'raw'
    data = reindex(data, format='%Y-%m-%d')

    # Optimization
    hopt_obj = HyperparameterOptimization(freq='D')
    opt_config = hopt_obj.run(data=data)

    if opt_config is None:
        raise Exception("FAILED: Unable to optimize hyper params")

    # Profiling
    de_obj = DataExploration(freq='D', **opt_config)
    training_data, pre_prc = de_obj.profile(data)

    # Identify Model
    model_class_name = opt_config['LuminaireModel']
    print(f'File: {input_file} Model: {model_class_name}')
    module = __import__('luminaire.model', fromlist=[''])
    model_class = getattr(module, model_class_name)

    # Training
    model_object = model_class(hyper_params=opt_config, freq='D')
    success, model_date, trained_model = model_object.train(data=training_data, **pre_prc)

    with open(output_path, 'wb') as output:
        pickle.dump(trained_model, output)
if __name__ == '__main__':
    import pdb;pdb.set_trace()
    save_structural('daily_test_time_series.csv', 'lad_filtering_model')
    save_structural('daily_test_time_series_seasonal.csv', 'lad_filtering_model_log_seasonal')
    save_structural('daily_test_time_series.csv', 'lad_structural_model')
    save_structural('daily_test_time_series_seasonal.csv', 'lad_structural_model_log_seasonal')
    save_window_density('window_density_test_hourly.csv', 'window_density_model')
    save_window_density('window_density_test_hourly.csv', 'window_density_model_hourly_aggregated')
    save_window_density('window_density_test_10_mins.csv', 'window_density_model_hourly_last_window')
