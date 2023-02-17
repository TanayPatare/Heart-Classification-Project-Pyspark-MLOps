import yaml
from Classification.exception import ClassificationException
import os,sys
from Classification.config.start_spark_session import spark_session
from config.dataframe_schema import schema
import dill


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

def load_dataset_for_prediction(data: list, columns:list):
    
    """
    Created to read a dataset in pyspark dataframe
    """
    try:
        pred_df = spark_session.createDataFrame(data, columns,schema=schema)
        return pred_df
    except Exception as e:
        raise ClassificationException(e,sys) from e

def save_object(file_path:str,obj):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise ClassificationException(e,sys) from e

def load_object(file_path:str):
    """
    file_path: str
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise ClassificationException(e,sys) from e

def write_yaml_file(file_path:str,data:dict=None):
    """
    Create yaml file 
    file_path: str
    data: dict
    """
    try:
        if os.path.exists(path=file_path):
            with open(file_path,"w") as yaml_file:
                if data is not None:
                    yaml.dump(data,yaml_file)
        else:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path,"w") as yaml_file:
                if data is not None:
                    yaml.dump(data,yaml_file)
    except Exception as e:
        raise ClassificationException(e,sys)