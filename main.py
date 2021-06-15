import time
from TS import TSPredict
from Config import Config


if __name__ == '__main__':

    c = Config()
    data = c.load_data()

    time_cost = {}

    tsa = TSPredict(data, freq = c.freq, future=c.future, insample=c.insample)

    # decompose model
    start = time.time()
    tsa.set_decompose_model(model_name=c.decompose_model, params=c.decompose_params)
    end = time.time()
    time_cost["decompose"] = end - start

    # trend model
    start = time.time()
    tsa.set_trend_model(model_name=c.trend_model, params=c.trend_params)
    tsa.predict_trend(params=c.trend_pred_params)
    end = time.time()
    time_cost["trend"] = end - start

    # seasonal model
    start = time.time()
    tsa.set_season_model(model_name=c.season_model, params=c.season_params)
    tsa.predict_season(params=c.trend_pred_params)
    end = time.time()
    time_cost["season"] = end - start

    # residual model
    is_white_noise = tsa.check_residual()
    if not is_white_noise:
        start = time.time()
        tsa.set_residual_model(model_name=c.residual_model, params=c.residual_params)
        tsa.predict_residual(params=c.residual_pred_params)
        end = time.time()
        time_cost["residual"] = end - start

    # final results
    tsa.predict(use_residual=not is_white_noise)
    time_cost["all"] = sum(time_cost.values())

    tsa.plot()
    for k, v in time_cost.items():
        print("Stage: {} ---- Time: {:.4f}s".format(k, v))

