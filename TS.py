import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, r2_score
from PredictionModel.ModelDispatcher import ModelDispatcher
from DecomposeModel.TSDecomposeModel import TSDecomposeModel


class TSPredict:
    def __init__(self, data, insample: bool = False):
        """
        Time series predictor class

        Parameters
        ----------
        data: pd.Series
            Time series data.
        insample: {bool, False}, optional
            in-sample predictions or out-of-sample predictions
        """
        self.TREND = "trend"
        self.SEASON = "season"
        self.RESIDUAL = "residual"

        self.__data = data
        self.__in_sample = insample

        self.__model_dispatcher = ModelDispatcher()
        self._tsDecomposeModel = TSDecomposeModel()

    def set_decompose_model(self,
                            model_name: str,
                            future: int = 0,
                            **kwargs):
        self.decompose_result = self._tsDecomposeModel.decompose(model_name, self.__data, kwargs)
        if self._tsDecomposeModel.get_model_name() == 'x11':
            self.half_period = self._tsDecomposeModel.x11_half_period()
        self.future = future
        if self.__in_sample:
            self.trend_train = self.decompose_result.trend[:-future]
            self.season_train = self.decompose_result.seasonal[:-future]
            self.residual_train = self.decompose_result.resid[:-future]
            # self.weights = self.decompose_result.weights[:-future]
            self.trend_test = self.decompose_result.trend[-future:]
            self.season_test = self.decompose_result.seasonal[-future:]
            self.residual_test = self.decompose_result.resid[-future:]
            # self.weights_test = self.decompose_result.weights[-future:]
            self.train = self.__data[:-future]
            self.test = self.__data[-future:]
        else:
            self.trend_train = self.decompose_result.trend
            self.season_train = self.decompose_result.seasonal
            self.residual_train = self.decompose_result.resid
            # self.weights = self.decompose_result.weights
            self.train = self.__data

    def set_trend_model(self, model_name: str, **kwargs):
        model = self.__model_dispatcher.dispatch(model_name, component=self.TREND)
        model.set_parameters(self.trend_train, kwargs)
        self.__trend_model = model

    def set_season_model(self, model_name: str, **kwargs):
        model = self.__model_dispatcher.dispatch(model_name, component=self.SEASON)
        model.set_parameters(self.season_train, kwargs)
        self.__season_model = model

    def set_residual_model(self, model_name, **kwargs):
        model = self.__model_dispatcher.dispatch(model_name, component=self.RESIDUAL)
        model.set_parameters(self.residual_train, kwargs)
        self.__residual_model = model

    def predict_trend(self, future: int = None, **kwargs):
        if self.__in_sample:
            future = self.future
            if self._tsDecomposeModel.is_x11():
                future += self.half_period
        else:
            assert (future)
        self.trend_predict = self.__trend_model.predict(future, kwargs)
        if self._tsDecomposeModel.is_x11():
            self.trend_predict = self.trend_predict[-(future - self.half_period):]

    def predict_season(self, **kwargs):
        if self.__in_sample:
            future = self.future
        else:
            future = kwargs.get("future")
        self.season_predict = self.__season_model.predict(future, kwargs)

    def predict_residual(self, **kwargs):
        if self.__in_sample:
            future = self.future
            if self._tsDecomposeModel.is_x11():
                future += self.half_period
        else:
            future = kwargs.get("future")
        self.residual_predict = self.__residual_model.predict(future, kwargs)
        if self._tsDecomposeModel.is_x11():
            self.residual_predict = self.residual_predict[-(future - self.half_period):]

    def predict(self, decompose_mode="add", use_residual=True):
        if decompose_mode == "add":
            self.total_predict = self.trend_predict + self.season_predict
            if use_residual:
                self.total_predict += self.residual_predict
        elif decompose_mode == "mul":
            self.total_predict = self.trend_predict * self.season_predict
            if use_residual:
                self.total_predict *= self.residual_predict
        else:
            raise Exception("decompose_mode must be `add` or `mul`")
        if self.__in_sample:
            decompose_model_name = self._tsDecomposeModel.get_model_name()
            if decompose_model_name == "x11":
                d = {"trend"   : [self.trend_test[self.half_period:], self.trend_predict[:-self.half_period]],
                     "season"  : [self.season_test, self.season_predict],
                     "residual": [self.residual_test[self.half_period:], self.residual_predict[:-self.half_period]],
                     "#total#" : [self.test, self.total_predict]
                     }
            else:
                d = {"trend"   : [self.trend_test, self.trend_predict],
                     "season"  : [self.season_test, self.season_predict],
                     "residual": [self.residual_test, self.residual_predict],
                     "#total#": [self.test, self.total_predict]
                     }
            for k, v in d.items():
                rmse = np.sqrt(mean_squared_error(v[0], v[1]))
                r2 = r2_score(v[0], v[1])
                adjust_r2 = 1 - ((1 - r2_score(v[0], v[1])) * (len(v[0]) - 1)) / (len(v[0]) - 2)
                print("RMSE for {}: {:.4f}".format(k, rmse))
                print("R2 for {}: {:.4f}".format(k, r2))
                print("Adjusted R2 for {}: {:.4f}".format(k, adjust_r2))
                print()

    def check_residual(self):
        # adf检验
        dftest = adfuller(self.residual_train.dropna())
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Obserfvisions Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)
        # 补充Box-Pierce检验或Ljung-Box检验,p值大于0.05时为白噪声,默认lag为1～40
        x = acorr_ljungbox(self.residual_train)
        white_noise = True
        for pv in x[1]:
            if pv < 0.05:
                white_noise = False
                break
        print("While noise: {}".format(white_noise))
        if white_noise:
            print("No need to model the residuals")
        else:
            print("Necessary to model the residuals")
        return white_noise

    def plot(self):
        self.decompose_result.plot()
        plt.show()
        plt.savefig("decompose.png")

        plt.clf()
        self.trend_train.plot(label="trend_train")
        self.trend_test.plot(label="trend_test")
        self.trend_predict.plot(label="trend_predict", alpha=0.5)
        plt.legend()
        plt.show()
        plt.savefig("trend.png")

        plt.clf()
        self.season_train.plot(label="season_train")
        self.season_test.plot(label="season_test")
        self.season_predict.plot(label="season_predict", alpha=0.5)
        plt.legend()
        plt.show()
        plt.savefig("season.png")

        plt.clf()
        self.residual_train.plot(label="residual_train")
        self.residual_test.plot(label="residual_test")
        self.residual_predict.plot(label="residual_predict", alpha=0.5)
        plt.legend()
        plt.show()
        plt.savefig("residual.png")

        plt.clf()
        self.train.plot(label="train")
        self.test.plot(label="test")
        self.total_predict.plot(label="predict", alpha=0.5)
        plt.legend()
        plt.show()
        plt.savefig("total.png")
