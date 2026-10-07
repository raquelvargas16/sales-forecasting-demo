import h2o
import os
from h2o.automl import H2OAutoML
from src.eng_sales_forecasting import preprocess_ts

class H2OAutoMLWrapper:
    def __init__(self, data_path: str="data/"):
        """
        Inicializa la clase H2OAutoMLWrapper.
        """
        if os.path.isfile(data_path):
            self.data_path = data_path
        else:
            self.data_path = "https://github.com/h2oai/h2o-tutorials/raw/master/h2o-world-2017/automl/data/powerplant_output.csv"

    def import_data(self):
        """
        Importa los datos desde la ruta especificada y si no se proporciona
        una ruta válida, descarga los datos desde un enlace predeterminado."""
        return h2o.import_file(self.data_path)
    def run_automl(self, data,
                   target: str="HourlyEnergyOutputMW",
                   test_ratio: float=0.2,
                   seed: int=42,
                   max_runtime_secs: int=60,
                   project_name = "powerplant_lb_frame"
                   ):
        splits = data.split_frame(ratios=[1-test_ratio], seed=seed)
        train = splits[0]
        test = splits[1]
        aml = H2OAutoML(max_runtime_secs = max_runtime_secs, seed = seed, project_name = project_name)
        aml.train(y = target, training_frame = train, leaderboard_frame = test)
        return aml
    def predict(self, aml_model, new_data):
        """
        Realiza predicciones utilizando el modelo ganador del AutoML.
        """
        return aml_model.leader.predict(new_data)

if __name__ == "__main__":
    h2o.init()
    # automl_wrapper = H2OAutoMLWrapper()
    # data = automl_wrapper.import_data()
    # aml_model = automl_wrapper.run_automl(data)
    # print(aml_model.leaderboard.head(15))

    ## Prueba con serie de tiempo
    ts_path = preprocess_ts()
    data_ts = h2o.import_file(path=ts_path)
    aml2 = H2OAutoML(max_runtime_secs = 60, seed = 42, project_name = "sales_forecasting")
    aml2.train(y = "Sales", training_frame = data_ts)
    print(aml2.leaderboard.sort(by="rmse").to_pretty_str())
    print(aml2.leader.model_performance(data_ts))
    