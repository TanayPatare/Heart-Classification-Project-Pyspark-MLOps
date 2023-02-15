from Classification.utils.util import read_yaml_file
from Classification.constant import *
from Classification.exception import ClassificationException
import sys
from Classification.utils.util import read_yaml_file
from Classification.logger import logging
import importlib
from collections import namedtuple


MetricInfoArtifact = namedtuple("MetricInfoArtifact",
                                ["model_name", "model_object","train_accuracy",
                                 "test_accuracy", "model_accuracy", "index_number"])

class ModelFactory:

    def __init__(self, model_config_path: str = None):
        try:
            logging.info(f"reading the model.yaml file [{self.config}]")
            self.config: dict = read_yaml_file(file_path=model_config_path)
        except Exception as e:
            raise ClassificationException(e,sys)


    def import_library_from_str(self,library,class_name):
        try:
            #importing library from string
            module = importlib.import_module(library)
            class_ref = getattr(module, class_name)
            return class_ref
        except Exception as e:
            raise ClassificationException(e,sys) from e


    def get_best_model(self,train_data):
        try:
            
            model_number = [MODULE_SELECTION_RF_KEY,MODULE_SELECTION_LR_KEY]
            List_of_models = []
            for i in model_number:

                #Reading first dictionary model
                model = self.config[MODULE_SELECTION_CLASS_KEY][i]
                
                #parameters for model
                Classifier_model_name = model[MODULE_CLASS_KEY]
                Classifier_model_library = model[MODULE_CLASS_MODULE_KEY]
                Classifier_model_params_features = model[MODULE_PARAMS_KEY][MODULE_PARAMS_FEATURECOL_KEY]
                Classifier_model_params_labelcol = model[MODULE_PARAMS_KEY][MODULE_PARAMS_LABELCOL_KEY]
                
                #importing library from string
                class_ref = self.import_library_from_str(Classifier_model_library,Classifier_model_name)
                
                # Creating Model
                model_name = class_ref(labelCol=Classifier_model_params_labelcol,featuresCol=Classifier_model_params_features)
                List_of_models.append(model_name.fit(train_data))

            return List_of_models

        except Exception as e:
            raise ClassificationException(e,sys) from e

    def evaluate_classification_model(self,model_list,train_dataset,test_dataset,base_accuracy):
        """
        Description:
        This function compare multiple regression model return best model
        Params:
        model_list: List of model
        train_dataset: Training dataset 
        test_dataset: Testing dataset
        base_accuracy: base accuracy for model 
        return
        It retured a named tuple
    
        MetricInfoArtifact = namedtuple("MetricInfoArtifact",
                                ["model_name", "model_object","train_accuracy",
                                 "test_accuracy", "model_accuracy", "index_number"])
        """
        try:
            index_number = 0
            metric_info_artifact = None
            for model in model_list:
                model_name = str(model)  #getting model name based on model object
                logging.info(f"{'>>'*30}Started evaluating model: [{type(model).__name__}] {'<<'*30}")
                
                #Getting prediction for training and testing dataset
                prediction_train = model.transform(train_dataset)
                prediction_test = model.transform(test_dataset)

                #parameters for evaluation
                models = self.config[MODULE_SELECTION_CLASS_KEY][MODULE_SELECTION_RF_KEY]
                modeleval_library = models[MODULE_EVAL_KEY]
                modeleval_name = models[MODULE_EVAL_CLASS_KEY]
                model_params_labelcol = models[MODULE_PARAMS_KEY][MODULE_PARAMS_LABELCOL_KEY]
                model_params_predictionCol = models[MODULE_PARAMS_KEY][MODULE_PARAMS_PREDICTIONCOL_KEY]
                model_evaluation_metric = models[MODULE_EVAL_METRIC_KEY][MODULE_EVAL_METRIC_CLASS_KEY]

                #importing library from string
                class_ref_eval = self.import_library_from_str(modeleval_library,modeleval_name)

                #Calculating accuracy for tarin and test dataset
                train_accuracy = class_ref_eval(labelCol=model_params_labelcol,
                    predictionCol=model_params_predictionCol,
                    metricName=model_evaluation_metric).evaluate(prediction_train)

                test_accuracy = class_ref_eval(labelCol=model_params_labelcol,
                    predictionCol=model_params_predictionCol,
                    metricName=model_evaluation_metric).evaluate(prediction_test)

                #Checking if model is generalised or not
                diff_test_train_acc = abs(test_accuracy - train_accuracy)
                model_accuracy = (2 * (train_accuracy * test_accuracy)) / (train_accuracy + test_accuracy)

                #logging all important metric
                logging.info(f"{'>>'*30} Score {'<<'*30}")
                logging.info(f"Train Score\t\t Test Score\t\t Difference in Score")
                logging.info(f"{train_accuracy}\t\t {test_accuracy}\t\t{diff_test_train_acc}")

                #if model accuracy is greater than base accuracy and train and test score is within certain thershold
                #we will accept that model as accepted model
                if model_accuracy >= base_accuracy and diff_test_train_acc < 0.05:
                    base_accuracy = model_accuracy
                    metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                                        model_object=model,
                                                        train_accuracy=train_accuracy,
                                                        test_accuracy=test_accuracy,
                                                        model_accuracy=model_accuracy,
                                                        index_number=index_number)
                    logging.info(f"Acceptable model found {metric_info_artifact}. ")
                index_number += 1
                if metric_info_artifact is None:
                    logging.info(f"No model found with higher accuracy than base accuracy")
                return metric_info_artifact

        except Exception as e:
            raise ClassificationException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 30}Model trainer log completed.{'<<' * 30} ")