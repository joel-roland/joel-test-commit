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

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
username_replaced = username.replace(".", "_").replace("@","_")
base_table_path = "dbfs:/home/"+username+"/"
table_location = base_table_path + 'boston_house_price'
table_name = username_replaced + ".boston_house_price"

# COMMAND ----------

  dbutils.fs.rm(table_location, recurse=True)

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {table_name}")

# COMMAND ----------

df.write.format('delta').save(table_location, mode='overwrite')

spark.sql(f"CREATE TABLE {table_name} USING DELTA LOCATION '{table_location}'")

# COMMAND ----------

displayHTML(f'<h3>Data is available at {table_name}</h3>')

# COMMAND ----------

