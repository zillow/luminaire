Introduction
========

.. image:: front.png
   :scale: 60%

Luminaire is an internal data quality service within Zillow Group that provides an ML driven solution for monitoring time series data. Luminaire provides several anomaly detection / forecasting capabilities that incorporates any correlational / seasonal pattern in the data over time and also incorporates the uncontrollable variations. Specifically, Luminaire is equipped with the following key features:

- **Generic Anomaly Detection Service:** Luminaire is a generic enough anomaly detection tool containing several classes of time series model focused towards catching any irregular fluctuations over different kinds of time series data.

- **Fully Automatic:** Luminaire performs optimization over different set of hyperparameters over several model classes and picks the optimal model for the time series under consideration. So, no model configuration is required from the user end.

- **Supports Diverse Anomaly Detection Types:** Luminaire supports different detection types:
 - Outlier Detection
 - Data Shift Detection
 - Trend Turning Detection
 - Null Data Detection
 - Density comparison for streaming data

Data Exploration and Profiling
-----------------------------------
Luminaire runs several exploratory testing on the data before sending it for the actual optimization or training. This step provides some batch insights about the raw training data on a given time window and also provides the flexibility of taking automated decisions regarding data pre-processing during the optimization process. Some of the tests and pre-processings includes:

 - Checking for recent data shifts.
 - Detect recent trend turnings.
 - Stationarity adjustments.
 - Imputation of Missing data.


Outlier Detection
-----------------------
Luminaire generates a model for a given time series based on it's recent patterns. Luminaire implements several modeling techniques to learn different variational patterns of the data that ranges from ARIMA, Filtering Models, Fourier Transform. Luminaire incorporates the global characteristics while learning the local patterns in order to make the learning process robust to any local fluctuations and for faster execution.

Configuration Optimization for Outlier Detection Models
-------------------------------------------------------------
Luminaire generates an idea on whether the time series shows exponential characteristics in terms of its variational patterns, whether holidays have any effects on the time series, whether the time series shows a long term correlational or Markovian pattern (depends on the last value only) etc. Luminaire uses hyperopt at it's core to optimize over the global hyperparameters for a given time series.

Anomaly Detection for Streaming Data
------------------------------------
Luminaire performs anomaly detection over streaming data by comparing volume density of the incoming data stream with a preset baseline time series window. Luminaire is capable of tracking time series windows over different data frequencies and is also autoconfigured to support most of the typical streaming use cases. 
