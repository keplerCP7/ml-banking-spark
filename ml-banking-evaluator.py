
# coding: utf-8

# In[1]:


from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

conf = SparkConf().setAppName("banking-linear-regression-analysis")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

df = sqlContext.read.parquet("s3://vl2-dlk/data-banking-demo/banking-output/")
df.registerTempTable("df_data")


# In[3]:


df.select('age', 'prediction').show(10)

