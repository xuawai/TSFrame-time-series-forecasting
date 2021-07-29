from datetime import datetime
import pandas as pd


class Config:

    def __init__(self):
        """
        Support two kinds of configurations: Use the decomposition model or not.
        Refer to self.decompose_example() and self.no_decompose_example() for the configuration.

        Parameters which must be set manually and correctly
        ----------
            insample: boolean
                in-sample predictions or out-of-sample predictions

            future: int
                the length of predictions

            decompose_model: str or None
                'x11', 'stl', 'robuststl'
                if set to None, use no decomposition mode.

            trend_model, season_model, residual_model, raw_model: str
                'holt', 'arima', 'prophet'

        Parameters which are highly recommended to be set
        ----------
            period : int
                The period of the input time series

        Parameters which can be set to an empty dict {}
        ----------
            decompose_params: {dict, None}
                - stl
                    {"period": {int, period}}

                - robuststl
                    {"period": {int, period}}

                - x11
                    {"period": {int, period}
                    "model": {str, "add" or "mul"}}

            trend_train_params, season_train_params, residual_train_params, raw_train_params: {dict, None}
                - holt
                    {"trend": {str, "add" or "mul"}
                    "seasonal": {str, "add" or "mul"}
                    "seasonal_periods": {int, period}}

                - arima
                    {"m": {int, period}}

                - prophet
                    {"growth": {str, "linear" or "logistic"},
                    "seasonality_mode": {str, "additive" or "multiplicative"}
                    "cap": {int, 10000}, "cap" can only work if "logistic"}

            trend_pred_params, season_pred_params, residual_pred_params, raw_pred_params: {dict, None}

        Example
        ----------
            self.decompose_example()
        or:
            self.no_decompose_example()
        """
        self.decompose_example()
        # self.no_decompose_example()

    def print_config(self):
        print("Configuration")
        print('*'*40)
        print("Basic Info:")
        print("insample = {}".format(self.insample))
        print("future = {}".format(self.future))
        print("period = {}".format(self.period))
        print("Model Info:")
        if self.decompose_model:
            print("decompose_model = {}".format(self.decompose_model))
            print("decompose_params = {}".format(self.decompose_params))
            print("trend_model = {}".format(self.trend_model))
            print("trend_train_params = {}".format(self.trend_train_params))
            print("season_model = {}".format(self.season_model))
            print("season_train_params = {}".format(self.season_train_params))
            print("residual_model = {}".format(self.residual_model))
            print("residual_train_params = {}".format(self.residual_train_params))
        else:
            print("raw_model = {}".format(self.raw_model))
            print("raw_train_params = {}".format(self.raw_train_params))
            print("raw_pred_params = {}".format(self.raw_pred_params))
        print('*' * 40)
        print()

    def no_decompose_example(self):
        self.insample = True
        self.future = 48
        self.period = 12

        self.decompose_model = None

        self.raw_model = 'arima'
        self.raw_train_params = {"m": self.period}
        self.raw_pred_params = {}

        # If use prophet, uncomment the following 3 lines.
        # self.raw_model = 'prophet'
        # self.raw_train_params = {"growth": "linear",
        #                          "seasonality_mode": "multiplicative",
        #                          "cap": 10000}
        # self.raw_pred_params = {}

        # If use holt, uncomment the following 3 lines.
        # self.raw_model = 'holt'
        # self.raw_train_params = {}
        # self.raw_pred_params = {}

    def decompose_example(self):
        self.insample = True
        self.future = 48
        self.period = 12

        self.decompose_model = 'x11'
        self.trend_model = 'holt'
        self.season_model = 'prophet'
        self.residual_model = 'arima'

        self.decompose_params = {"period": self.period,
                                 "model": "add"}
        self.trend_train_params = {"trend": "add",
                             "seasonal": "add",
                             "seasonal_periods": self.period
                             }
        self.season_train_params = {"growth": "linear",
                                    "seasonality_mode": "multiplicative",
                                    "cap": 10000}
        self.residual_train_params = {"m": self.period}

        self.trend_pred_params = {}
        self.season_pred_params = {}
        self.residual_pred_params = {}

    def load_data(self):
        def security_data():
            date_parse = lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.000+0800')
            data_orig = pd.read_csv('security.csv',
                                    index_col='_time',  # 指定索引列
                                    parse_dates=['_time'],  # 将指定列按照日期格式来解析
                                    date_parser=date_parse  # 日期格式解析器
                                    )
            data_orig = data_orig[~data_orig.index.duplicated(keep="last")]
            t_index = pd.date_range(data_orig.index[0], data_orig.index[-1], freq='min')
            data_orig.reindex(t_index)
            data = data_orig["value"]
            return data

        def air_passenger():
            date_parse = lambda x: datetime.strptime(x, '%Y-%m')
            data = pd.read_csv('AirPassengers.csv',
                               index_col='Month',  # 指定索引列
                               parse_dates=['Month'],  # 将指定列按照日期格式来解析
                               date_parser=date_parse  # 日期格式解析器
                               )
            data = data["#Passengers"]
            return data
        return air_passenger()
