import sys

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import pandas as pd

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel

conf = SparkConf().setAppName("banking-linear-regression-Inference")
#sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

df_test = sqlContext.read.parquet("s3://vl2-dlk/data-banking-demo/banking-input/test")

model = LogisticRegressionModel.load("s3://vl2-dlk/data-banking-demo/Modelo")



df_prediction = model.transform(df_test)

df_prediction.write.mode("Append").parquet("s3://vl2-dlk/data-banking-demo/banking-output")
