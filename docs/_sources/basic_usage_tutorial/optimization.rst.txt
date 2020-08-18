Configuration Optimization
==========================

Luminaire *HyperparameterOptimization* performs auto-selection of the best data preprocessing configuration and the outlier detection model training configuration with respect to the input time series. This option enables Luminaire to work as a hands-off system where the user only has to provide the input data along with its frequency. This option should be used if the user wants avoid any manual configuration and should be called prior to the data pre-processing and training steps.

>>> from luminaire.optimization.hyperparameter_optimization import HyperparameterOptimization
>>> print(data)
               raw
index              
2020-01-01  1326.0
2020-01-02  1552.0
2020-01-03  1432.0
2020-01-04  1470.0
2020-01-05  1565.0
...             ...
2020-06-03  1934.0
2020-06-04  1873.0
2020-06-05  1674.0
2020-06-06  1747.0
2020-06-07  1782.0
>>> hopt_obj = HyperparameterOptimization(freq='D')
>>> opt_config = hopt_obj.run(data=data)
>>> print(opt_config)
{'LuminaireModel': 'LADStructuralModel', 'data_shift_truncate': 0, 'fill_rate': 0.742353444620679, 'include_holidays_exog': 1, 'is_log_transformed': 1, 'max_ft_freq': 2, 'p': 1, 'q': 1}

Fully Automatic Outlier Detection
-------------------------------------

Since the optimized configuration contains all the parameters required for data pre-processing and training, this can be used downstream for performing the data pre-processing and training.

>>> from luminaire.exploration.data_exploration import DataExploration
>>> de_obj = DataExploration(freq='D', **opt_config)
>>> training_data, pre_prc = de_obj.profile(data)
>>> print(training_data)
                raw  interpolated
2020-01-01  1326.0      7.190676
2020-01-02  1552.0      7.347943
2020-01-03  1432.0      7.267525
2020-01-04  1470.0      7.293697
2020-01-05  1565.0      7.356279
...            ...           ...
2020-06-03  1934.0      7.567862
2020-06-04  1873.0      7.535830
2020-06-05  1674.0      7.423568
2020-06-06  1747.0      7.466227
2020-06-07  1782.0      7.486052

The above piece of code makes the data ready to be ingested for training. The only step left before training is to extract the luminaire outlier detection model object for the optimized configuration.

>>> model_class_name = opt_config['LuminaireModel']
>>> module = __import__('luminaire.model', fromlist=[''])
>>> model_class = getattr(module, model_class_name)
>>> print(model_class)
<class 'luminaire_models.model.lad_structural.LADStructuralModel'>

Since, we have to optimal model class along with other optimal configurations, we can run training as follows:

>>> model_object = model_class(hyper_params=opt_config, freq='D')
>>> success, model_date, trained_model = model_object.train(data=training_data, **pre_prc)
>>> print(success, model_date, trained_model)
(True, '2020-06-07 00:00:00', <luminaire_models.model.lad_structural.LADStructuralModel object at 0x7fe2b47a7978>)

This trained model is now ready to be used for scoring future data points.

>>> trained_model.score(2000, '2020-06-08')
{'Success': True, 'IsLogTransformed': 1, 'LogTransformedAdjustedActual': 7.601402334583733, 'LogTransformedPrediction': 7.529710533463001, 'LogTransformedStdErr': 0.06217883425408564, 'LogTransformedCILower': 7.422390543346913, 'LogTransformedCIUpper': 7.62662106869458, 'AdjustedActual': 2000.000000000015, 'Prediction': 1861.566274906425, 'StdErr': 110.9167321105633, 'CILower': 1672.028177505716, 'CIUpper': 2051.104372307134, 'ConfLevel': 90.0, 'ExogenousHolidays': 0, 'IsAnomaly': False, 'IsAnomalyExtreme': False, 'AnomalyProbability': 0.7545715087682185, 'DownAnomalyProbability': 0.12271424561589073, 'UpAnomalyProbability': 0.8772857543841093, 'ModelFreshness': 0.1}


