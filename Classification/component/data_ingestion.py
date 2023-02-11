from Classification.entity.config_entity import DataIngestionConfig
from Classification.entity.artifact_entity import DataIngestionArtifact
import os, sys
from Classification.logger import logging
from Classification.exception import ClassificationException
import tarfile 
from six.moves import urllib
from Classification.config.start_spark_session import spark_session as sp
from pyspark.sql.types import *
from config.dataframe_schema import schema
from pyspark.ml.feature import *
from zipfile import ZipFile
import pandas as pd

class DataIngestion:

    def __init__(self, data_ingestion_config : DataIngestionConfig):
        try:
            logging.info(f"{'='*20}Data Ingestion log Started{'='*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise ClassificationException(e,sys) from e

    def download_heart_data(self) -> str:
        try:
            #downloading data from external source
            download_url = self.data_ingestion_config.dataset_download_url

            #folder location to download
            rar_download_dir = self.data_ingestion_config.rar_download_dir
            
            #creating folder for rar download
            if os.path.exists(rar_download_dir):
                os.remove(rar_download_dir)

            os.makedirs(rar_download_dir,exist_ok=True)

            #extracting file name from url
            classifiaction_rar_file_name = os.path.basename(download_url)

            #creating directory path
            rar_file_path = os.path.join(rar_download_dir,classifiaction_rar_file_name)

            logging.info(f"Downloading file from :[{download_url}] into [{rar_file_path}]")
            
            #downloading data
            urllib.request.urlretrieve(download_url,rar_file_path)
            
            logging.info(f"File [{rar_file_path}] has been downloaded succesfully")
            
            return rar_file_path

        except Exception as e:
            raise ClassificationException(e,sys) from e


    def extract_rar_file(self,rar_file_path:str):
        try:
            #folder location to download
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            #creating folder for rar download
            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)
            
            os.makedirs(raw_data_dir,exist_ok=True)
            
            logging.info(f"Extracting rar file:[{rar_file_path}] into dir [{raw_data_dir}]")
            
            #extracting files from raw_data_dir
            with ZipFile(rar_file_path) as classification_rar_file_obj:
                classification_rar_file_obj.extractall(path = raw_data_dir)

            logging.info(f"Extraction completed")

        except Exception as e:
            raise ClassificationException(e,sys) from e


    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            
            #getting csv file name
            csv_file_name = os.listdir(raw_data_dir)[0]

            #creating directory
            csv_file_path = os.path.join(raw_data_dir,csv_file_name)
            
            #reading Data
            logging.info(f"reading csv files: [{csv_file_path}]")
            path = csv_file_path
            Classification_data_frame = sp.read.csv(path=path,schema=schema,header=True)

            #Spliting File into training and testing
            logging.info(f"Spliting File into training and testing")
            train_dataset = None
            test_dataset = None
            (train_dataset, test_dataset) = Classification_data_frame.randomSplit([0.8, 0.2])

            #Creating directory for train and test data
            train_dir = os.path.join(self.data_ingestion_config.ingested_train_dirr,csv_file_name)
            test_dir = os.path.join(self.data_ingestion_config.ingested_test_dirr,csv_file_name)

            #creating folder for train and test
            if train_dataset is not None:
                os.makedirs(train_dir,exist_ok=True)
                logging.info(f"dumping train data to its respective folder : [{train_dir}]")
                #dumping train data to its respective folder
                train_dataset.write.csv(path=train_dir)

            if test_dataset is not None:
                os.makedirs(test_dir,exist_ok=True)
                logging.info(f"dumping test data to its respective folder : [{test_dir}]")
                #dumping test data to its respective folder
                test_dataset.write.csv(path=test_dir)
           
            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_dir,
                                                            test_filepath= test_dir,
                                                            message=f"Data ingestion completed  Succesfully",
                                                            is_ingested=True)
            logging.info(f"DataIngestionArtifact: [{DataIngestionArtifact}]")
            return data_ingestion_artifact

        except Exception as e:
            raise ClassificationException(e,sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            rar_file_path = self.download_heart_data()
            self.extract_rar_file(rar_file_path)
            return self.split_data_as_train_test()
        except Exception as e:
            raise ClassificationException(e,sys) from e
    
    def __del__(self):
         logging.info(f"{'='*20}Data Ingestion log completed{'='*20}")