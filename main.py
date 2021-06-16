import time
from TS import TSPredict
from Config import Config


if __name__ == '__main__':

    c = Config()
    data = c.load_data()
    time_cost = {}

    tsa = TSPredict(data, future=c.future, insample=c.insample)
    if c.decompose_model:
        # decompose model
        start = time.time()
        tsa.set_decompose_model(model_name=c.decompose_model, params=c.decompose_params)
        end = time.time()
        time_cost["decompose"] = end - start

        # trend model
        start = time.time()
        tsa.set_trend_model(model_name=c.trend_model, train_params=c.trend_train_params)
        tsa.predict_trend(pred_params=c.trend_pred_params)
        end = time.time()
        time_cost["trend"] = end - start

        # seasonal model
        start = time.time()
        tsa.set_season_model(model_name=c.season_model, train_params=c.season_train_params)
        tsa.predict_season(pred_params=c.trend_pred_params)
        end = time.time()
        time_cost["season"] = end - start

        # residual model
        is_white_noise = tsa.check_residual()
        if not is_white_noise:
            start = time.time()
            tsa.set_residual_model(model_name=c.residual_model, train_params=c.residual_train_params)
            tsa.predict_residual(pred_params=c.residual_pred_params)
            end = time.time()
            time_cost["residual"] = end - start

        # final results
        tsa.predict(decompose_mode=c.decompose_params["model"] if "model" in c.decompose_params else "add", use_residual=not is_white_noise)
        time_cost["all"] = sum(time_cost.values())

        # plot
        tsa.plot()

        # print the configuration
        c.print_config()

        # print the running time
        for k, v in time_cost.items():
            print("Stage: {} ---- Time: {:.4f}s".format(k, v))
    else:
        #train & predict
        start = time.time()
        tsa.ts_predict_with_no_decompose(model_name=c.raw_model, train_params=c.raw_train_params, pred_params=c.raw_pred_params)
        end = time.time()

        # plot
        tsa.plot()

        # print the configuration
        c.print_config()

        # print the running time
        print("Stage: {} ---- Time: {:.4f}s".format("raw", end-start))

