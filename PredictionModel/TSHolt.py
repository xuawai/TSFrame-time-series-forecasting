from PredictionModel.BasePredictionModel import BasePredictionModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class TSHolt(BasePredictionModel):

    def __init__(self, model_name):
        super().__init__(model_name)

    def set_parameters(self, train_data, params):
        self.model = ExponentialSmoothing(endog=train_data,
                                            trend=params.get("trend", "add"),
                                            seasonal=params.get("seasonal", "add"),
                                            seasonal_periods = params.get("seasonal_periods", None)
                                            ).fit()

    def predict(self, future, params):
        prediction = self.model.forecast(future)
        return prediction
