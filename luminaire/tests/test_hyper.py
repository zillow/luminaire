from luminaire.optimization.hyperparameter_optimization import *

class TestHyperparameterOptimization(object):

    def test_run1(self, test_data_with_missing):
        """Test using the default random_state=None"""
        hyper_obj = HyperparameterOptimization(freq='D', detection_type='OutlierDetection')
        hyper_parameters = hyper_obj.run(test_data_with_missing, max_evals=5)

        assert isinstance(hyper_parameters, dict)

    def test_run2(self, test_data_with_missing):
        """Test defining a random_state"""
        hyper_obj = HyperparameterOptimization(freq='D', detection_type='OutlierDetection', random_state=42)
        hyper_parameters = hyper_obj.run(test_data_with_missing, max_evals=5)

        assert isinstance(hyper_parameters, dict)
