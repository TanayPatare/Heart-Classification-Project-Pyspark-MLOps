import yaml
from Classification.exception import ClassificationException
import os,sys
from Classification.config.start_spark_session import spark_session
from config.dataframe_schema import schema


def read_yaml_file(file_path:str):

    """
    Reads a yaml file and returns the contents as dictionary.
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise ClassificationException(e,sys) from e

def load_dataset(train_file_path:str, test_file_path:str):
    
    """
    Created to read a dataset in pyspark dataframe
    """
    try:
        train_df = spark_session.read.csv(train_file_path,schema=schema,header=True)
        test_df = spark_session.read.csv(test_file_path,schema=schema,header=True)
        return train_df, test_df
    except Exception as e:
        raise ClassificationException(e,sys) from e