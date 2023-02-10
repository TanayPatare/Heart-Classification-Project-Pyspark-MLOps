from pyspark.sql import SparkSession
import os, sys

spark_session = SparkSession.builder.master('local[*]').appName('Heart_prediction').getOrCreate()
