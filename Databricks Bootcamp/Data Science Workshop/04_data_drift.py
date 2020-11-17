# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC #Versioning data and detecting data drift
# MAGIC 
# MAGIC Once your model is in production, you will be using it to score new data coming through and it is likely that data will change over time and your model will no longer be accurate

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Let's plot the RMSE on the predictions table

# COMMAND ----------

prediction_table_name = '/demo/ML/boston_house_price_prediction'
boston_house_price_prediction_stream = spark.readStream.format("delta").load(prediction_table_name)
boston_house_price_prediction_stream.createOrReplaceTempView("boston_house_price_prediction_stream")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select TS, power(avg(power(PRICE - PREDICTION, 2)), 0.5) as rmse 
# MAGIC from boston_house_price_prediction_stream
# MAGIC GROUP BY TS
# MAGIC ORDER BY TS

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Checkout the Dashboard View
# MAGIC 
# MAGIC You can use create a streaming dashboard using the native dashboarding function

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Now we simulate data drift
# MAGIC 
# MAGIC We will be running a number of update statement to the dataset and we will be scoring them after each one. We should see that the streaming dashboard update as we append new prediction data.

# COMMAND ----------

# MAGIC %md
# MAGIC Run the scoring again before updating the data and we can see that RMSE stays the same

# COMMAND ----------

model_id =  # replace with one of the models from your experiment

# COMMAND ----------

model = mlflow.sklearn.load_model("runs:/" + model_id + "/model")
df = spark.sql('select * from boston_house_price').toPandas()

pred = score_dataset(df, model)

spark.createDataFrame(pred).write.format('delta').save("/demo/ML/boston_house_price_prediction", mode='append')

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC We will manually update a few columns to mimick a data change and we will rescore the dataset to see how it impacts the RMSE

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE boston_house_price SET PTRATIO = 0.0 WHERE PTRATIO >= 0.0;

# COMMAND ----------

model = mlflow.sklearn.load_model("runs:/" + model_id + "/model")
df = spark.sql('select * from boston_house_price').toPandas()

pred = score_dataset(df, model)

spark.createDataFrame(pred).write.format('delta').save("/demo/ML/boston_house_price_prediction", mode='append')

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE boston_house_price SET B = 0.0 WHERE B >= 0.0;

# COMMAND ----------

model = mlflow.sklearn.load_model("runs:/" + model_id + "/model")
df = spark.sql('select * from boston_house_price').toPandas()

pred = score_dataset(df, model)

spark.createDataFrame(pred).write.format('delta').save("/demo/ML/boston_house_price_prediction", mode='append')

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE boston_house_price SET LSTAT = 0.0 WHERE LSTAT >= 0.0;

# COMMAND ----------

model = mlflow.sklearn.load_model("runs:/" + model_id + "/model")
df = spark.sql('select * from boston_house_price').toPandas()

pred = score_dataset(df, model)

spark.createDataFrame(pred).write.format('delta').save("/demo/ML/boston_house_price_prediction", mode='append')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We can see that RMSE is getting worse, let's retrain the model to improve prediction accuracy

# COMMAND ----------

algo = xgb.XGBRegressor

params = {
  'pca_params': {'n_components': 11},
  'algo_params': {'max_depth': 4, 'learning_rate': 0.11, 'n_estimators': 1000, 'early_stopping_rounds': 100, 'objective': 'reg:squarederror'}
}

run_name = "XGBoost Retrain"
experiment_id = 2736989290285991
TABLE_NAME = 'boston_house_price'

FEATURES = [
  'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
]

df_train, df_test, version = prep_data(TABLE_NAME)
run_info, model_stats = run_algo(df_train, df_test, version, FEATURES, algo, params, experiment_id, run_name=run_name, do_shap=True)

# COMMAND ----------

model = mlflow.sklearn.load_model("runs:/" + run_info.run_uuid + "/model")
df = spark.sql('select * from boston_house_price').toPandas()
pred = score_dataset(df, model)

spark.createDataFrame(pred).write.format('delta').save("/demo/ML/boston_house_price_prediction", mode='append')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Now we can see that our RMSE is back to a reasonable value using the new model.