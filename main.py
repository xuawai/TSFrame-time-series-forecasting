import time
from datetime import datetime
import pandas as pd
from TS import TSPredict


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

if __name__ == '__main__':

    def trial():
        """
        These parameters have default values..

        Parameters
        ----------
        stl_config = {"period": [12]}
        robuststl_config = {"period": [12]}
        x11_config = {"period": [12],
                      "model": ["add", "mul"]}
        holt_config = {"trend": ["add", "mul"],
                       "seasonal": ["add", "mul"],
                       "seasonal_periods": [12]
                        }
        arima_config = {"m": [12]}
        prophet_config = {"growth": ["linear", "logistic"],
                      "cap": [None]}   # `cap` can be set if `logistic`


        These parameters must be set manually and correctly..
        Parameters
        ----------
        freqs = ['D', 'W', 'M', 'MS', ...],
            Any valid frequency for pd.date_range,
            Refer to https://blog.csdn.net/wangqi_qiangku/article/details/79384731
        """
        data = air_passenger()

        # parameters
        period = 12
        freq = "MS"

        time_cost = {}
        start = time.time()
        tsa = TSPredict(data, insample=True)
        # decompose model
        tsa.set_decompose_model("X11", future=36, period=period)
        time_cost["decompose"] = time.time() - start

        # trend model
        start = time.time()
        tsa.set_trend_model("prophet", period=period)
        tsa.predict_trend(freq=freq)
        time_cost["trend"] = time.time() - start - time_cost["decompose"]

        # seasonal model
        start = time.time()
        tsa.set_season_model("holt", seasonal_periods=period)
        tsa.predict_season(freq=freq)
        time_cost["season"] = time.time() - start - time_cost["decompose"] - time_cost["trend"]

        # residual model
        start = time.time()
        is_white_noise = tsa.check_residual()
        if not is_white_noise:
            tsa.set_residual_model("arima", m=period)
            tsa.predict_residual(freq=freq)
            time_cost["residual"] = time.time() - start - time_cost["decompose"] - time_cost["trend"] - time_cost["season"]

        # final results
        tsa.predict(use_residual=not is_white_noise)
        time_cost["all"] = time.time() - start

        tsa.plot()
        for k, v in time_cost.items():
            print("Stage: {} ---- Time: {}s".format(k, v))

    trial()
