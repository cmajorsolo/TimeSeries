# Notes
Weekly and daily data:
The seasonal period for weekly data is normally 52(A year contains 52 weeks). Most of the methods cannot handle such large seasonal period efficiently.
Daily data is more complex because it not only contains a large seasonal period but also contains multiple seasonal patterns. Single seasonal models like seasonal ARIMA is applicable to a time series which is relatively short so that one type of seasonality is present. When the series is long enough so that some of the longer seasonal periods become apparent, it will be necessary to use STL, dynamic harmonic regressions or TBATS[1]. However, all these models can’t deal with covariants like moving events. The only choice is a dynamic regression model, where the predictors include any dummy holiday effects (and possibly also the seasonality using Fourier terms)[1].
SARIMA
Is built for univariate dataset has trend or seasonality[2]. To check if the SARIMA model can be applied to a dataset, verify if there is seasonality(S), AR&MA component(value at time t depends on value at t-n), trend(I). The hyperparameters for SARIMA is (p, d, q)(P, D, Q)m. P, D, Q are the analogs for p, d, q except for the seasonal component m.
Question: It it required to use log transformation on a series that has an exponential trend and seasonality before modeling in SARIMA? Ans: Yes if your data has a changing variance after trend and seasonality is removed, you can fix it with a box cox or similar power transform[2].
Question: Why my forecasted time series one step behind the actual time series? Ans: It means that your model is making a persistence forecast. This is a forecast where the input to the forecast is predicted as the output. If a sophisticated model, such as a neural network, is outputting a persistence forecast, it might mean: 1. That the model requires further tuning. 2. That the chosen model cannot address your specific dataset. 3. It might also mean that your time series problem is not predictable. [3]
Baseline model:
A baseline model is used as a reference point for other models. If the model performs same or worse than the baseline model, it should be fixed or abandoned. A baseline should be simple, fast and repeatable. The most common baseline method for supervised machine learning is the Zero Rule algorithm.This algorithm predicts the majority class in the case of classification, or the average outcome in the case of regression. This could be used for time series, but does not respect the serial correlation structure in time series datasets. The equivalent technique for use with time series dataset is the persistence algorithm.The persistence algorithm uses the value at the previous time step (t-1) to predict the expected outcome at the next time step (t+1)[4].
Grid search:
Grid search is a method of Hyperparameter optimization. It is an exhaustive searching through a manually specified subset of hyperparameter space of a learning algorithm. A grid search algorithm must be guided by some performance metric, typically measured by cross-validation on the training set. [5]
Steps for timeseries forecasting:
Build baseline model.
Feature selection (e.g. n day differenciation)
Build hyperparameter space.
Grid search
Define the model
Fit the model
Test the model (cross validation sklearn.model_selection.TimeSeriesSplit[6])
Compare the result with the baseline model
Prophet Algorithm from Facebook
It is an algorithm to build forecasting models for time series data. Unlike the traditional approach, it tries to fit additive regression models a.k.a. ‘curve fitting’[7].
Comparing Prophet with ARIMA performance on Bitcoin data
Data: daily data from year 2016 to 2018. Data before 2016 is truncated off because it is quite stable
Build a correlation matrix between GBP, EUR, JPY
Train, validation and test dataset size is decided by R square, MAPE and RMSPE values
Backtest:
It is common practice to fit a model using training data, and then to evaluate its performance on a test data set. The way this is usually done means the comparisons on the test data use different forecast horizons. The forecast variance usually increases with the forecast horizon, so if we are simply averaging the absolute or squared errors from the test set, we are combining results with different variances.[9]
The general cross validation is not applicable in time-series data because the coherence of the data is broken during the splitting train-test dataset process. Walk forward is a more robust technique in which the data is split in a time sequence.
[1]https://otexts.com/fpp2/weekly.html
[2]https://machinelearningmastery.com/sarima-for-time-series-forecasting-in-python/
[3]https://machinelearningmastery.com/faq/single-faq/why-is-my-forecasted-time-series-right-behind-the-actual-time-series/
[4]https://machinelearningmastery.com/persistence-time-series-forecasting-with-python/
[5]https://en.wikipedia.org/wiki/Hyperparameter_optimization
[6]https://towardsdatascience.com/time-series-modeling-using-scikit-pandas-and-numpy-682e3b8db8d1
[7]https://blog.exploratory.io/an-introduction-to-time-series-forecasting-with-prophet-package-in-exploratory-129ed0c12112
[8]https://www.researchgate.net/publication/329400738_Bitcoin_Forecasting_Using_ARIMA_and_PROPHET
[9]Forecasting: Principles and Practice https://otexts.com/fpp2/forecasting-on-training-and-test-sets.html

