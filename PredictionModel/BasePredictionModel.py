from abc import abstractmethod
import pandas as pd


class BasePredictionModel():

    def __init__(self, model_name):
        self.__model_name = model_name
        self.model = None

    @abstractmethod
    def set_parameters(self, train_data:pd.Series, params:dict):
        pass

    @abstractmethod
    def predict(self, future:int, params:dict):
        pass