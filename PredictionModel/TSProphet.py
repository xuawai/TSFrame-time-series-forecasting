from PredictionModel.BasePredictionModel import BasePredictionModel
import numpy as np
import pandas as pd
from fbprophet import Prophet


class TSProphet(BasePredictionModel):

    def __init__(self, model_name):
        super().__init__(model_name)

    def set_parameters(self, train_data, params):
        data = pd.DataFrame(train_data.copy())
        data['y_orig'] = data['y'] = train_data
        data['ds'] = list(train_data.index)
        data['cap'] = params.get("cap", 10000)
        self.data = data
        self.train_data = train_data
        self.model = Prophet(growth=params.get("growth", "linear"),
                            holidays=params.get("holidays", None)).fit(data)

    def predict(self, future, params):
        freq = self.train_data.index.inferred_freq
        future_data = self.model.make_future_dataframe(
            periods=future, freq=freq)
        future_data['cap'] = self.data['cap']
        prediction = self.model.predict(
            future_data)['yhat'][-future:]
        prediction = pd.Series(data=prediction.values,
                                       index=future_data.ds.values[-future:])
        return prediction