from Classification.utils.util import read_yaml_file
from Classification.constant import *
from Classification.exception import ClassificationException
import sys
from Classification.utils.util import read_yaml_file
from Classification.logger import logging
import importlib

MODEL_SELECTION_KEY = 'model_selection'

class ModelFactory:

    def __init__(self, model_config_path: str = None):
        try:
            logging.info(f"reading the model.yaml file [{self.config}]")
            self.config: dict = read_yaml_file(file_path=model_config_path)
            self.models_initialization_config: dict = dict(self.config[MODEL_SELECTION_KEY])
            self.initialized_model_list = None
        except Exception as e:
            raise ClassificationException(e,sys)

    def get_best_model(self,train_data,base_accuracy:int):
        try:
            self.train_data = train_data
            self.base_accuracy = base_accuracy
            
            model_number = [MODULE_SELECTION_RF_KEY,MODULE_SELECTION_LR_KEY]
            List_of_models = []
            for i in model_number:
                #Reading first dictionary model
                model = self.config[MODULE_SELECTION_CLASS_KEY][i]
                # Creating Model
                Classifier_model_name = model[MODULE_CLASS_KEY]
                Classifier_model_library = model[MODULE_CLASS_MODULE_KEY]
                Classifier_model_params_features = model[MODULE_PARAMS_KEY][MODULE_PARAMS_FEATURECOL_KEY]
                Classifier_model_params_labelcol = model[MODULE_PARAMS_KEY][MODULE_PARAMS_LABELCOL_KEY]
                module = importlib.import_module(Classifier_model_library)
                class_ref = getattr(module, Classifier_model_name)
                model_name = class_ref(labelCol=Classifier_model_params_labelcol,featuresCol=Classifier_model_params_features)
                List_of_models.append(model_name.fit(self.train_data))

            return List_of_models

        except Exception as e:
            raise ClassificationException(e,sys) from e