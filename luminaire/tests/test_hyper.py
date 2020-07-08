from luminaire.optimization.hyperparameter_optimization import *

class TestHyperparameterOptimization(object):

    def test_run(self, test_data_with_missing):

        hyper_obj = HyperparameterOptimization(freq='D', detection_type='OutlierDetection')
        hyper_parameters = hyper_obj.run(test_data_with_missing, max_evals=5)

        assert isinstance(hyper_parameters, dict)
