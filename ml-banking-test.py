
# coding: utf-8

# In[ ]:


from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel

conf = SparkConf().setAppName("banking-linear-regression-Test")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

df_test = sqlContext.read.parquet("s3://vl2-dlk/data-banking-demo/banking-input/test")


# In[ ]:


model = LogisticRegressionModel.load("s3://vl2-dlk/data-banking-demo/Modelo")

df_prediction = model.transform(df_test)


# In[ ]:


df_prediction.write.mode("Append").parquet("s3://vl2-dlk/data-banking-demo/banking-output")

