import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fbprophet import Prophet
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import DecomposeResult, STL, seasonal_decompose

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox

class TSDecomposeModel:

    def __init__(self):
        self.model_name = None
        pass

    def decompose(self, model_name, train_data, params):
        model_name = model_name.lower()
        if model_name == "stl":
            result = STL(train_data, period=params.get("period", None)).fit()
        elif model_name == "robuststl" or model_name == "rstl":
            result = STL(train_data, robust=True, period=params.get("period", None)).fit()
        elif model_name == "x11":
            result = seasonal_decompose(train_data,
                                     period=params.get("period", None),
                                     model=params.get("model", "additive"),
                                     extrapolate_trend=params.get("period", 1))
        else:
            raise Exception("Decompose Model {} has not been registered!".format(model_name))
        self.result = result
        self.model_name = model_name
        return result

    def x11_half_period(self):
        assert self.model_name == 'x11'
        self.result._trend = self.result.trend.dropna()
        self.result._seasonal = self.result.seasonal.dropna()
        self.result._resid = self.result.resid.dropna()
        half_period = (len(self.result._seasonal) - len(self.result._trend)) // 2
        return half_period

    def is_x11(self):
        return self.model_name == "x11"

    def get_model_name(self):
        return self.model_name
