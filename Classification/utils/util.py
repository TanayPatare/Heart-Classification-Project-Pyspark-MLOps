import yaml
from Classification.exception import ClassificationException
import os,sys



def read_yaml_file(file_path:str):

    """
    Reads a yaml file and returns the contents as dictionary.
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise ClassificationException(e,sys) from e