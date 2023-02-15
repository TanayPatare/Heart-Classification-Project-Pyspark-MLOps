from Classification.entity.config_entity import ModelTrainerConfig
from Classification.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from Classification.logger import logging
from Classification.exception import ClassificationException
import sys
from Classification.entity.model_factory import ModelFactory
from Classification.utils.util import load_dataset, save_object
from Classification.entity.model_factory import MetricInfoArtifact
from pyspark.ml.feature import *
from Classification.constant import *

class ClassifiactionEstimatorModel:
    def __init__(self,trained_model_object):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_object: trained_model_object
        """
        self.trained_model_object = trained_model_object
    
    def data_transformation(self,data):
        feature = VectorAssembler(inputCols = data.columns[:len(data.columns)-1],outputCol=FEATURE_COLUMN)
        feature_vector= feature.transform(data)
        feature_vector_select = feature_vector.select([FEATURE_COLUMN,TARGET_COLUMN])
        return feature_vector_select

    def predict(self, X):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which gurantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        transformed_feature = self.data_transformation(X)
        return self.trained_model_object.predict(transformed_feature)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"

class ModelTrainer:
    
    def __init__(self,model_trainer_config: ModelTrainerConfig, 
                    data_transformation_artifact: DataTransformationArtifact
                    ):

        try:
            logging.info(f"{'='*20}Model Trainer log Started{'='*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise ClassificationException(e,sys) from e
    
     
    def initiate_model_trainer(self) ->ModelTrainerArtifact:
        try:
            logging.info(f"Loading transformed training dataset")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            train_data = load_dataset(file_path=transformed_train_file_path)

            logging.info(f"Loading transformed testing dataset")
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path
            test_data= load_dataset(file_path=transformed_test_file_path)

            logging.info(f"Extracting model config file path")
            model_config_file_path = self.model_trainer_config.model_config_file_path

            logging.info(f"Initializing model factory class using above model config file: {model_config_file_path}")
            model_factory = ModelFactory(model_config_path=model_config_file_path)

            base_accuracy = self.model_trainer_config.base_accuracy
            logging.info(f"Expected accuracy: {base_accuracy}")

            logging.info(f"Initiating operation model selecttion")
            best_model = model_factory.get_best_model(train_data = train_data)
            logging.info(f"Best model found on training dataset: {best_model}")

            metric_info:MetricInfoArtifact = model_factory.evaluate_classification_model(
                                    model_list=best_model,train_dataset = train_data,
                                    test_dataset = test_data,base_accuracy=base_accuracy
                                    )
            if metric_info is not None:
                logging.info(f"Best found model on both training and testing dataset.")
            else:
                logging.info(f"No model found on both training and testing dataset.")

            model_object = metric_info.model_object
            trained_model_file_path=self.model_trainer_config.trained_model_file_path
            classification_model = ClassifiactionEstimatorModel(trained_model_object=model_object)
            logging.info(f"Saving model at path: {trained_model_file_path}")
            save_object(file_path=trained_model_file_path,obj=classification_model)

            model_trainer_artifact=  ModelTrainerArtifact(
                            is_trained=True,message="Model Trained successfully",
                            trained_model_file_path=trained_model_file_path,
                            train_accuracy=metric_info.train_accuracy,
                            test_accuracy=metric_info.test_accuracy,
                            model_accuracy=metric_info.model_accuracy
            )
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise ClassificationException(e,sys) from e