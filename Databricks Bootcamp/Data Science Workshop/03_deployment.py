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

df = spark.sql('select * from boston_house_price').toPandas()
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

# MAGIC %sql 
# MAGIC select *, boston_ml_model(CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT) as prediction
# MAGIC from boston_house_price 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Writing out predictions to Delta

# COMMAND ----------

prediction_table_name = '/demo/ML/boston_house_price_prediction'

# COMMAND ----------

# MAGIC %fs rm -r /demo/ML/boston_house_price_prediction

# COMMAND ----------

model = mlflow.sklearn.load_model("runs:/" + winning_model_id + "/model")
df = spark.sql('select * from boston_house_price').toPandas()

pred = score_dataset(df, model)

spark.createDataFrame(pred).write.format('delta').save("/demo/ML/boston_house_price_prediction", mode='overwrite')

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC DROP TABLE IF EXISTS boston_house_price_prediction;
# MAGIC 
# MAGIC CREATE TABLE boston_house_price_prediction
# MAGIC USING delta
# MAGIC LOCATION '/demo/ML/boston_house_price_prediction';

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC SELECT * FROM boston_house_price_prediction

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Now that we have deployed our model, go to the next [notebook]($./04_data_drift) to see what happens when data change overtime.