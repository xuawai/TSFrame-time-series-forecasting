import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, r2_score
from PredictionModel.ModelDispatcher import ModelDispatcher
from DecomposeModel.TSDecomposeModel import TSDecomposeModel


class TSPredict:
    def __init__(self, data: pd.Series, future: int, insample: bool):
        self.RAW = "raw"
        self.TREND = "trend"
        self.SEASON = "season"
        self.RESIDUAL = "residual"

        self.future = future
        self.data = data
        self.insample = insample

        self.model_dispatcher = ModelDispatcher()
        self.ts_decompose_model = TSDecomposeModel()

        self.is_decompose = True

    def set_decompose_model(self, model_name: str, params: dict):
        self.decompose_result = self.ts_decompose_model.decompose(model_name, self.data, params)
        if self.insample:
            self.trend_train = self.decompose_result.trend[:-self.future]
            self.season_train = self.decompose_result.seasonal[:-self.future]
            self.residual_train = self.decompose_result.resid[:-self.future]
            self.trend_test = self.decompose_result.trend[-self.future:]
            self.season_test = self.decompose_result.seasonal[-self.future:]
            self.residual_test = self.decompose_result.resid[-self.future:]
            self.train = self.data[:-self.future]
            self.test = self.data[-self.future:]
        else:
            self.trend_train = self.decompose_result.trend
            self.season_train = self.decompose_result.seasonal
            self.residual_train = self.decompose_result.resid
            self.train = self.data

    def set_trend_model(self, model_name: str, train_params: dict):
        self.trend_model = self.model_dispatcher.dispatch(model_name, component=self.TREND)
        self.trend_model.set_parameters(self.trend_train, train_params)

    def set_season_model(self, model_name: str, train_params: dict):
        self.season_model = self.model_dispatcher.dispatch(model_name, component=self.SEASON)
        self.season_model.set_parameters(self.season_train, train_params)

    def set_residual_model(self, model_name:str, train_params: dict):
        self.residual_model = self.model_dispatcher.dispatch(model_name, component=self.RESIDUAL)
        self.residual_model.set_parameters(self.residual_train, train_params)

    def predict_trend(self, pred_params: dict):
        self.trend_predict = self.trend_model.predict(self.future, pred_params)

    def predict_season(self, pred_params: dict):
        self.season_predict = self.season_model.predict(self.future, pred_params)

    def predict_residual(self, pred_params: dict):
        self.residual_predict = self.residual_model.predict(self.future, pred_params)

    def eval(self, name: str, v1: pd.Series, v2: pd.Series):
        rmse = np.sqrt(mean_squared_error(v1, v2))
        r2 = r2_score(v1, v2)
        adjust_r2 = 1 - ((1 - r2_score(v1, v2)) * (len(v1) - 1)) / (len(v2) - 2)
        print("RMSE for {}: {:.4f}".format(name, rmse))
        print("R2 for {}: {:.4f}".format(name, r2))
        print("Adjusted R2 for {}: {:.4f}".format(name, adjust_r2))

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
        print("\nEvaluation")
        print("*" * 40)
        if self.insample:
            d = {self.TREND : [self.trend_test, self.trend_predict],
                 self.SEASON : [self.season_test, self.season_predict],
                 self.RESIDUAL : [self.residual_test, self.residual_predict],
                 self.RAW: [self.test, self.total_predict]
                 }
            for k, v in d.items():
                self.eval(name=k, v1=v[0], v2=v[1])
        else:
            print("Out-of-sample predictions. There is no RMSE or R-squared.")
        print("*" * 40)
        print()

    def check_residual(self):
        print("\nStatistical Test")
        print("*" * 40)
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
        print("white_noise = {}".format(white_noise))
        if white_noise:
            print("No need to model the residuals")
        else:
            print("Need to model the residuals")
        print("*" * 40)
        return white_noise

    def ts_predict_with_no_decompose(self, model_name: str, train_params: dict, pred_params: dict):
        """
        use a single model for TS without any decomposition models.
        """
        self.is_decompose = False
        if self.insample:
            self.train = self.data[:-self.future]
            self.test = self.data[-self.future:]
        else:
            self.train = self.data
        model = self.model_dispatcher.dispatch(model_name, component=self.RAW)
        model.set_parameters(self.train, train_params)
        self.no_decompose_predict = model.predict(self.future, pred_params)
        print("\nEvaluation")
        print("*" * 40)
        if self.insample:
            self.eval(name=self.RAW, v1=self.test, v2=self.no_decompose_predict)
        else:
            print("Out-of-sample predictions. There is no RMSE or R-squared.")
        print("*" * 40)
        print()

    def plot(self):
        if self.is_decompose:
            self.decompose_result.plot()
            plt.show()
            plt.savefig("decompose.png")

            plt.clf()
            self.trend_train.plot(label="trend_train")
            if self.insample:
                self.trend_test.plot(label="trend_test")
            self.trend_predict.plot(label="trend_predict", alpha=0.5)
            plt.legend()
            plt.show()
            plt.savefig("trend.png")

            plt.clf()
            self.season_train.plot(label="season_train")
            if self.insample:
                self.season_test.plot(label="season_test")
            self.season_predict.plot(label="season_predict", alpha=0.5)
            plt.legend()
            plt.show()
            plt.savefig("season.png")

            plt.clf()
            self.residual_train.plot(label="residual_train")
            if self.insample:
                self.residual_test.plot(label="residual_test")
            self.residual_predict.plot(label="residual_predict", alpha=0.5)
            plt.legend()
            plt.show()
            plt.savefig("residual.png")

            plt.clf()
            self.train.plot(label="train")
            if self.insample:
                self.test.plot(label="test")
            self.total_predict.plot(label="predict", alpha=0.5)
            plt.legend()
            plt.show()
            plt.savefig("total.png")
        else:
            self.train.plot(label="train")
            if self.insample:
                self.test.plot(label="test")
            self.no_decompose_predict.plot(label="predict", alpha=0.5)
            plt.legend()
            plt.show()
            plt.savefig("no_decompose_total.png")

