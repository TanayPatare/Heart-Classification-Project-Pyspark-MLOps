from pyspark.sql import SparkSession
import os, sys
from Classification.logger import logging
from Classification.exception import ClassificationException

try:
    logging.info(f"Starting Spark Session")
    spark_session = SparkSession.builder.master('local[*]').appName('Heart_prediction').getOrCreate()
except Exception as e:
    raise ClassificationException(e,sys) from e