from Classification.entity.config_entity import ModelTrainerConfig
from Classification.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from Classification.logger import logging
from Classification.exception import ClassificationException
from Classification.constant import FEATURE_COLUMN, TARGET_COLUMN
import sys
from Classification.entity.model_factory import ModelFactory
from Classification.utils.util import load_dataset
class Model_trainer:
    
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
            best_model = model_factory.get_best_model(data = train_data, base_accuracy=base_accuracy)
            logging.info(f"Best model found on training dataset: {best_model}")

            
        except Exception as e:
            raise ClassificationException(e,sys) from e