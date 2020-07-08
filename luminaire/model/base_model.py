# -*- coding: utf-8 -*-
import pickle
from typing import Tuple


class BaseModelHyperParams(object):
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.params = {**kwargs}


class BaseModelObject(object):
    @classmethod
    def save(cls, model_object):
        return pickle.dumps(model_object)

    @classmethod
    def load(cls, model_output: bytes):
        return pickle.loads(model_output)


class BaseModel(object):
    """
    This is the base class for all models. Do not use this for production tasks. 
    """
    import abc as _abc
    __version__ = "2.0"

    def __init__(self, **kwargs):
        self._params = kwargs

    def __call__(self, *args, **kwargs):
        self._run(*args, **kwargs)

    @_abc.abstractmethod
    def train(self, data, **kwargs) -> Tuple[bool, BaseModelObject]:
        pass

    @classmethod
    @_abc.abstractmethod
    def score(cls, data, pred_date, model: BaseModelObject, **kwargs):
        pass

    @_abc.abstractmethod
    def _run(self, data, *args, **kwargs):
        """
        This is the main function where the model will be called. The subclass method must follow the same function
        signature.

        :param data: input data to a single model. May be a (pickled) pandas series/dataframe, a list, or even a single
            number depending on your model's needs. This *cannot* be any spark object.
        :param args: more data, if needed, to be passed into the model
        :param kwargs: model parameters. These are taken from the config when the model object is instantiated, and
            are not passed in as the model is called.
        :return: results from the model. This must be a dictionary where the keys are string types, and the values are
            simple json/yaml-serializable types. Nested dictionaries are *untested*.
        :rtype: list[dict[str, any]]
        """
        return data

    def get_info(self):
        """
        This function should return the model configuration to be stored in the "model attributes" column of the
        output table.

        :return: the model's basic information plus
        :rtype: dict
        """
        info = {"ModelName": self.__class__.__name__,
                "Version"  : self.__class__.__version__}
        info.update(self._config)
        return info

    def run(self, data, *cols):
        """
        Runs model on each row of the data.

        :param pyspark.sql.DataFrame data: spark dataframe with one row per model.
        :param cols: column name(s) to run model on.
        :return: a spark dataframe
        """
        import pyspark.sql.functions as F
        import pyspark.sql.types as T
        from datetime import date, datetime

        def json_serialize(obj):
            """
            JSON serializer for objects not serializable by default json code
            This function currently only handles datetime and date objects

            :param obj: Object to serialize
            :return: json serialized object
            """
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            raise TypeError("Type %s not serializable" % type(obj))

        def _run(*inp):
            """
            Function to call the model _run function

            :param tuple inp: inputs passed to the function.
                TimeSeries Example:
                ([Row(index=datetime.datetime(2019, 1, 1, 0, 0), raw=1197387.0, interpolated=1197387.0),
                Row(index=datetime.datetime(2019, 1, 2, 0, 0), raw=1449210.0, interpolated=1449210.0), ... ],
                Row(_MetricName=u'injections', email_routing_domain=u'att.net'),
                datetime.datetime(2019, 3, 1, 16, 30))

            :return: model result - list of tuples e.g. [("{}", "{}", Timestamp), ("{}", "{}", Timestamp), ...]
            """
            import json
            output = self._run(*inp)

            if isinstance(output, list):
                output = [(json.dumps(model_attribute, default=json_serialize),
                           json.dumps(model_result, default=json_serialize),
                           data_date)
                          for model_attribute, model_result, data_date in output]
            elif isinstance(output, tuple):
                output = [(json.dumps(output[0], default=json_serialize),
                           json.dumps(output[1], default=json_serialize),
                           output[2])]
            return output

        run_udf = F.udf(_run, T.ArrayType(
            T.StructType([T.StructField('model_attributes', T.StringType()),
                          T.StructField('model_results', T.StringType()),
                          T.StructField('data_date', T.TimestampType())
                          ])))

        new_df = (data.withColumn('model_output', run_udf(*cols)))

        return new_df
