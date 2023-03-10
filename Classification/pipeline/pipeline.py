from Classification.config.configuration import Configuration
from Classification.logger import logging
from Classification.exception import ClassificationException
from Classification.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact,\
                                                    DataTransformationArtifact, ModelTrainerArtifact,\
                                                        ModelEvaluationArtifact
from Classification.component.data_ingestion import DataIngestion
from Classification.component.data_validation import DataValidation
from Classification.component.data_transformation import DataTransformation
from Classification.component.model_trainer import ModelTrainer
from Classification.component.model_evaluation import ModelEvaluation
import sys

class Pipeline:
    
    def __init__(self, config:Configuration = Configuration())-> None:
        try:
            self.config = config
            pass
        except Exception as e:
            raise ClassificationException(e,sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config= self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise ClassificationException(e,sys) from e

    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config= self.config.get_data_validation_config(),
                                            data_ingestion_artifact=data_ingestion_artifact)
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise ClassificationException(e,sys) from e

    def start_data_transformation(self,data_ingestion_artifact:DataIngestionArtifact,
                                data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(data_transformation_config = self.config.get_data_transformation_config,
                                                    data_ingestion_artifact = data_ingestion_artifact,
                                                    data_validation_artifact = data_validation_artifact)
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise ClassificationException(e,sys) from e

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(model_trainer_config=self.config.get_model_trainer_config(),
                                         data_transformation_artifact=data_transformation_artifact
                                         )
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise ClassificationException(e, sys) from e
    def start_model_evaluation(self, 
                               model_trainer_artifact: ModelTrainerArtifact
                               ) -> ModelEvaluationArtifact:
        try:
            model_eval = ModelEvaluation(
                model_evaluation_config=self.config.get_model_evaluation_config(),
                model_trainer_artifact=model_trainer_artifact
                )
            return model_eval.initiate_model_evaluation()
        except Exception as e:
            raise ClassificationException(e, sys) from e
    def start_model_pusher(self):
        pass

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                                                                    data_validation_artifact=data_validation_artifact,
                                                                    model_trainer_artifact=model_trainer_artifact)
            if  model_evaluation_artifact.is_model_accepted:
                model_pusher_artifact = self.start_model_pusher(model_eval_artifact=model_evaluation_artifact)
                logging.info(f'Model pusher artifact: {model_pusher_artifact}')
            else:
                logging.info("Trained model rejected.")
            logging.info("Pipeline completed.")      

        except Exception as e:
            raise ClassificationException(e,sys) from e

    def run(self):
        try:
            self.run_pipeline()
        except Exception as e:
            raise e