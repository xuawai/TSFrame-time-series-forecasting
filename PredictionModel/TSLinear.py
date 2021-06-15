from PredictionModel.BasePredictionModel import BasePredictionModel
from sklearn.linear_model import LinearRegression
import numpy as np


class TSLinear(BasePredictionModel):

    def __init__(self, model_name):
        super().__init__(model_name)

    def set_parameters(self, train_data, params):
        self.train_data = train_data
        lin = LinearRegression()
        self.model = lin.fit(np.arange(len(train_data.values)).reshape(-1, 1),
                             train_data.values)

    def predict(self, future:int, freq, params):
        prediction = self.model.predict(
                np.arange(len(self.train_data.values),
                          len(self.train_data.values) + future).reshape(-1, 1))
        return prediction
