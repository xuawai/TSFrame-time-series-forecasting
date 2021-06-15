from datetime import datetime
import pandas as pd


class Config:

    def __init__(self):
        """
        Parameters which must be set manually and correctly
        ----------
            insample: boolean
                in-sample predictions or out-of-sample predictions

            future: int
                the length of predictions

            freqs : {str, 'D"}
                'D', 'W', 'M', 'MS'...
                Any valid frequency for pd.date_range,
                Refer to https://blog.csdn.net/wangqi_qiangku/article/details/79384731

            decompose_model: str
                'x11', 'stl', 'robuststl'

            trend_model, season_model, residual_model: str
                'holt', 'arima', 'prophet'

        Parameters which are highly recommended to be set
        ----------
            period : int
                The period of the input time series

        Parameters with optional values
        ----------
            docompose_params: {dict, None}
                stl
                    {"period": {int, period}}

                robuststl
                    {"period": {int, period}}

                x11
                    {"period": {int, period}
                    "model": {str, "add" or "mul"}}

            trend_params, season_params, residual_params: {dict, None}
                holt
                    {"trend": {str, "add" or "mul"}
                    "seasonal": {str, "add" or "mul"}
                    "seasonal_periods": {int, period}}

                arima
                    {"m": {int, period}}

                prophet
                    {"growth": {str, "linear" or "logistic"}
                    "cap": {int, None}, "cap" can only be set if "logistic"}

            trend_pred_params, season_pred_params, residual_params: {dict, None}
        """
        self.insample = True
        self.future = 36
        self.period = 12
        self.freq = "MS"

        self.decompose_model = 'x11'
        self.trend_model = 'prophet'
        self.season_model = 'holt'
        self.residual_model = 'arima'

        self.docompose_params = {"period": self.period,
                                 "model": "add"}
        self.trend_params = {"trend": "add",
                             "seasonal": "add",
                             "seasonal_periods": self.period
                             }
        self.sanson_params = {"growth": "linear",
                              "cap": None}
        self.residual_params = {"m": self.period}

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
