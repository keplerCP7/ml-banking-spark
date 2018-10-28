import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

conf = SparkConf().setAppName("banking-linear-regression-Training")
#sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

df = sqlContext.read.parquet("s3://vl2-dlk/data-banking-demo/banking-input/")
df.registerTempTable("df_data_set")

train, test = df.randomSplit([0.7, 0.3], seed = 2018)

test.write.mode("Append").parquet("s3://vl2-dlk/data-banking-demo/banking-input/test")

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
lrModel = lr.fit(train)

lrModel.write().overwrite().save("s3://vl2-dlk/data-banking-demo/Modelo/")
