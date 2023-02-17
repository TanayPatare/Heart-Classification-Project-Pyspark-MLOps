from Classification.exception import ClassificationException
from Classification.utils.util import load_object
import sys,os

class HeartClassifier:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise ClassificationException(e, sys) from e


    def predict(self, X):
        try:
            model = load_object(file_path=self.model_dir)
            value = model.predict(X)
            return value
        except Exception as e:
            raise ClassificationException(e, sys) from e
