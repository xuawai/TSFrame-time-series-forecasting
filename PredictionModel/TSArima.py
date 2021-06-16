from PredictionModel.BasePredictionModel import BasePredictionModel
from pmdarima import auto_arima
import pandas as pd


class TSArima(BasePredictionModel):

    def __init__(self, model_name):
        super().__init__(model_name)

    def set_parameters(self, train_data, params):
        self.train_data = train_data
        self.model = auto_arima(train_data,
                                m = params.get("m", 1),
                                seasonal=params.get("seasonal", True),
                                trace=params.get("trace", True),
                                error_action=params.get("error_action", 'ignore'),
                                suppress_warnings=params.get("suppress_warnings", True)).fit(train_data)

    def predict(self, future, params):
        freq = self.train_data.index.inferred_freq
        future_data = pd.date_range(start=self.train_data.index[-1], periods=future + 1, freq=freq)[1:]
        prediction = self.model.predict(n_periods=future)
        prediction = pd.Series(data=prediction, index=future_data)
        return prediction