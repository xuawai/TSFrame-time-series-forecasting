import time
from TS import TSPredict
from Config import Config


if __name__ == '__main__':

    c = Config()
    data = c.load_data()

    time_cost = {}
    start = time.time()
    tsa = TSPredict(data, freq = c.freq, future=c.future, insample=c.insample)

    # decompose model
    tsa.set_decompose_model(model_name=c.decompose_model, params=c.docompose_params)
    time_cost["decompose"] = time.time() - start

    # trend model
    start = time.time()
    tsa.set_trend_model(model_name=c.trend_model, params=c.trend_params)
    tsa.predict_trend(params=c.trend_pred_params)
    time_cost["trend"] = time.time() - start - time_cost["decompose"]

    # seasonal model
    start = time.time()
    tsa.set_season_model(model_name=c.season_model, params=c.sanson_params)
    tsa.predict_season(params=c.trend_pred_params)
    time_cost["season"] = time.time() - start - time_cost["decompose"] - time_cost["trend"]

    # residual model
    start = time.time()
    is_white_noise = tsa.check_residual()
    if not is_white_noise:
        tsa.set_residual_model(model_name=c.residual_model, params=c.residual_params)
        tsa.predict_residual(params=c.residual_pred_params)
        time_cost["residual"] = time.time() - start - time_cost["decompose"] - time_cost["trend"] - time_cost["season"]

    # final results
    tsa.predict(use_residual=not is_white_noise)
    time_cost["all"] = time.time() - start

    tsa.plot()
    for k, v in time_cost.items():
        print("Stage: {} ---- Time: {}s".format(k, v))

