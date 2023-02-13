from Classification.exception import ClassificationException
from Classification.logger import logging
from Classification.entity.config_entity import DataTransformationConfig
from Classification.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact,DataTransformationArtifact
import os, sys
from Classification.utils.util import load_dataset
from pyspark.ml.feature import *


class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                        data_ingestion_artifact: DataIngestionArtifact,
                        data_validation_artifact: DataValidationArtifact) -> None:
        try:
            logging.info(f"{'='*20}Data Transformation log Started{'='*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise ClassificationException(e,sys) from e

    def get_transformed_train_test_data(self,data):
        try:
            feature = VectorAssembler(inputCols = data.columns[:len(data.columns)-1],outputCol="features")
            feature_vector= feature.transform(data)
            feature_vector_select = feature_vector.select(['features','target'])
            return feature_vector_select
        except Exception as e:
            raise ClassificationException(e,sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info(f"Obtaining preprocessing object.")
            logging.info(f"Obtaining training and test file path.")
            train_file_path=self.data_ingestion_artifact.train_file_path,
            test_file_path=self.data_ingestion_artifact.test_filepath

            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df, test_df = load_dataset(
                train_file_path=train_file_path,
                test_file_path=test_file_path
            )

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            transformed_train_df = self.get_transformed_train_test_data(data = train_df)
            transformed_test_df = self.get_transformed_train_test_data(data = test_df)

            transformed_train_dir = self.data_transformation_config.transform_train_dir
            transformed_test_dir = self.data_transformation_config.transform_test_dir

            train_file_name = os.path.basename(train_file_path)
            test_file_name = os.path.basename(test_file_path)

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)
            
            logging.info(f"Saving transformed training and testing dataframe.")
            #dumping train data to its respective folder
            transformed_train_df.write.csv(path=transformed_train_file_path)
            #dumping test data to its respective folder
            transformed_test_df.write.csv(path=transformed_test_file_path)

            preprocessed_object_file_path = None

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            transformed_train_file_path=transformed_train_file_path,
            transformed_test_file_path=transformed_test_file_path,
            preprocessed_object_file_path= preprocessed_object_file_path
            )
            logging.info(f"Data Transformation Pipeline Completed")
            return data_transformation_artifact
        except Exception as e:
            raise ClassificationException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30}Data Transformation log completed.{'<<'*30} \n\n")