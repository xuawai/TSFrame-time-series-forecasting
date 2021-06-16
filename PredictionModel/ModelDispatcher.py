from PredictionModel.TSHolt import TSHolt
from PredictionModel.TSArima import TSArima
from PredictionModel.TSProphet import TSProphet
from PredictionModel.TSLinear import TSLinear

class ModelDispatcher:

    def __init__(self,):
        self.component2modelname = {}
        self.component2model = {}

    def get_model(self, component):
        assert component in self.component2model
        return self.component2model[component]

    def get_model_name(self, component):
        assert component in self.component2modelname
        return self.component2modelname[component]

    def dispatch(self, model_name, component):
        model_name = model_name.lower()
        if model_name == "holt":
            model = TSHolt(model_name)
        elif model_name == "arima":
            model = TSArima(model_name)
        elif model_name == "prophet":
            model = TSProphet(model_name)
        elif model_name == "linear":
            model = TSLinear(model_name)
        else:
            raise Exception("Model #{}# has not been registered!".format(model_name))
        assert component in ["trend", "season", "residual", "raw"]
        self.component2modelname[component] = model_name
        self.component2model[component] = model
        return model



