import pickle
import pandas as pd
from pathlib import Path
from luminaire.optimization.hyperparameter_optimization import HyperparameterOptimization
from luminaire.exploration.data_exploration import DataExploration
import logging

logging.basicConfig(level=logging.DEBUG)

def save(input_file, output_file):
    import pdb;pdb.set_trace()
    input_folder =Path(__file__).parent.joinpath('luminaire', 'tests','datasets')
    output_folder =Path(__file__).parent.joinpath('luminaire', 'tests','models')
    
    input_file = input_folder.joinpath(input_file)
    output_file = output_folder.joinpath(output_file)

    print(f"Input File: {input_file}")
    print(f"Output File: {output_file}")

    data = pd.read_csv(input_file)
    data['index'] = pd.to_datetime(data['index'], format='%Y-%m-%d')
    # Input data should have a time column set as the index column of the dataframe and a value column named as 'raw'

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
    module = __import__('luminaire.model', fromlist=[''])
    model_class = getattr(module, model_class_name)

    # Training
    model_object = model_class(hyper_params=opt_config, freq='D')
    success, model_date, trained_model = model_object.train(data=training_data, **pre_prc)

    with open(output_file, 'wb') as output:
        pickle.dump(trained_model, output)

if __name__ == '__main__':
    save('daily_test_time_series.csv', 'lad_filtering_model')