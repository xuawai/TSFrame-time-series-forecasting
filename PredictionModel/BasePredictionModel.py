from abc import ABC, abstractmethod


class BasePredictionModel():

    def __init__(self, model_name):
        self.__model_name = model_name
        self.model = None

    @abstractmethod
    def set_parameters(self, train_data, params):
        pass