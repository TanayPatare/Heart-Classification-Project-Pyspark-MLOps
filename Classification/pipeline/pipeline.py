from Classification.config.configuration import Configuration
from Classification.logger import logging
from Classification.exception import ClassificationException
from Classification.entity.artifact_entity import DataIngestionArtifact
from Classification.entity.config_entity import DataIngestionConfig
from Classification.component.data_ingestion import DataIngestion
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

    def start_data_validation(self):
        pass

    def start_data_transformation(self):
        pass

    def start_model_trainer(self):
        pass

    def start_model_pusher(self):
        pass

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
        except Exception as e:
            raise ClassificationException(e,sys) from e