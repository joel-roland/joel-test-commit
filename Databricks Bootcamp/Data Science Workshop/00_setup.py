# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC #Setup
# MAGIC 
# MAGIC This will write the Boston Property Dataset to Delta as the starting point for the rest of the Notebooks

# COMMAND ----------

from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def load_data():
  _boston_dataset = load_boston()
  X = _boston_dataset['data']
  y = _boston_dataset['target']
  
  columns = list(_boston_dataset['feature_names']) + ['PRICE']
  df = pd.DataFrame(np.hstack([X, y.reshape(y.shape[0], 1)]), columns=columns)
  
  return spark.createDataFrame(df)

df = load_data()

# COMMAND ----------

# MAGIC %fs rm -r /demo/ML/boston_house_price

# COMMAND ----------

table_name = '/demo/ML/boston_house_price'

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS boston_house_price")

# COMMAND ----------

df.write.format('delta').save(table_name, mode='overwrite')

spark.sql(f"CREATE TABLE boston_house_price USING DELTA LOCATION '{table_name}'")

# COMMAND ----------

displayHTML(f'<h3>Data is available at {table_name}</h3>')

# COMMAND ----------

