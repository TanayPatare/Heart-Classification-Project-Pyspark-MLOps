# schema of Dataset
from Classification.config.start_spark_session import spark_session as sp
from pyspark.sql.types import *

schema = StructType([
    StructField("age", IntegerType(),nullable=True),
    StructField("sex", IntegerType(),nullable=True),
    StructField("cp", IntegerType(),nullable=True),
    StructField("trestbps", IntegerType(),nullable=True),
    StructField("chol", IntegerType(),nullable=True),
    StructField("fbs", IntegerType(),nullable=True),
    StructField("restecg", IntegerType(),nullable=True),
    StructField("thalach", IntegerType(),nullable=True),
    StructField("exang", IntegerType(),nullable=True),
    StructField("oldpeak", FloatType(),nullable=True),
    StructField("slope", IntegerType(),nullable=True),
    StructField("ca", IntegerType(),nullable=True),
    StructField("thal", IntegerType(),nullable=True),
    StructField("target", IntegerType(),nullable=True),
])