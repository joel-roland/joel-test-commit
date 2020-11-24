# Databricks notebook source
# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %md
# MAGIC ## Operationalising Model
# MAGIC 
# MAGIC There are a number of different ways we can deploy the model.

# COMMAND ----------

# DBTITLE 1,Input the ID of the model that you want to deploy
winning_model_id =  # replace with one of the models from your experiment

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Native Python
# MAGIC 
# MAGIC Use `mlflow.sklearn.load_model` to load the model and do batch prediction

# COMMAND ----------

df = spark.sql(f'select * from {table_name}').toPandas()
display(df)

# COMMAND ----------

import mlflow.sklearn
model = mlflow.sklearn.load_model("runs:/" + winning_model_id + "/model")
model.predict_pd(df)[:10]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Spark UDF
# MAGIC 
# MAGIC Use `mlflow.pyfunc.spark_udf` to register model as a SparkSQL UDF and just run it on SQL data

# COMMAND ----------

spark.udf.register(
  'boston_ml_model', 
  mlflow.pyfunc.spark_udf(spark, "runs:/" + winning_model_id + "/model")
)

# COMMAND ----------

display(spark.sql(f"select *, boston_ml_model(CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT) as prediction from {table_name}"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Writing out predictions to Delta

# COMMAND ----------

dbutils.fs.rm(prediction_table_location, recurse=True)

# COMMAND ----------

model = mlflow.sklearn.load_model("runs:/" + winning_model_id + "/model")
df = spark.sql(f"select * from {table_name}").toPandas()

pred = score_dataset(df, model)

spark.createDataFrame(pred).write.format('delta').save(prediction_table_location, mode='overwrite')

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {prediction_table_name}")

# COMMAND ----------

spark.sql(f"CREATE TABLE {prediction_table_name} USING DELTA LOCATION '{prediction_table_location}'")

# COMMAND ----------

display(spark.sql(f'select * from {prediction_table_name}'))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Now that we have deployed our model, go to the next [notebook]($./04_data_drift) to see what happens when data change overtime.