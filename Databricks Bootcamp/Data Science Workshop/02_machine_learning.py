# Databricks notebook source
# MAGIC %md 
# MAGIC ## Training ML models with Databricks & Mlflow
# MAGIC We will:
# MAGIC - Train some Machine Learning models with scikit-learn and XGBoost
# MAGIC - Get some custom performance visualisations with MatplotLib 
# MAGIC - Do manual hyperparameter tuning

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Run the utils notebook
# MAGIC 
# MAGIC The notebook contains the ML wrapper function `run_algo` that we will be using throughout this notebook. 
# MAGIC 
# MAGIC At a high level it does the following steps:
# MAGIC 1. Train the model using the train dataset
# MAGIC 2. Score the train and test dataset
# MAGIC 3. Run Model Diagnostics
# MAGIC 4. Log the parameters, metrics, artifacts and model into MLFlow

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,Look at the Dataset
# MAGIC %sql
# MAGIC 
# MAGIC select * from boston_house_price

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Create your MLFlow Experiment
# MAGIC 
# MAGIC Copy and paste the experiment ID to the cell below

# COMMAND ----------

experiment_id =  # get experiment ID from Experiment UI you created

displayHTML(f"<h2>Make sure you can see your experiment on <a href='#mlflow/experiments/{experiment_id}'>#mlflow/experiments/{experiment_id}</a></h2>")

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Define some global variables

# COMMAND ----------

TABLE_NAME = 'boston_house_price'

FEATURES = [
  'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
]

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Using Elastic Net as a baseline algorithm
# MAGIC 
# MAGIC Elastic net is a regularized regression method that linearly combines the L1 and L2 penalties of the lasso and ridge methods (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)

# COMMAND ----------

algo = ElasticNet

params = {
  'pca_params': {'n_components': 12},
  'algo_params': {'alpha': 0.05, 'l1_ratio': 0.1, 'normalize': False}
}

run_name = "Elastic Net"

df_train, df_test, version = prep_data(TABLE_NAME)
run_info, model_stats = run_algo(df_train, df_test, version, FEATURES, algo, params, experiment_id, run_name=run_name, do_shap=True)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### When in doubt, use XGBoost
# MAGIC 
# MAGIC XGBoost is an implementation of gradient boosted decision trees designed for speed and performance. It has been dominating applied machine learning and Kaggle competitions for structured or tabular data. (https://xgboost.readthedocs.io/en/latest/index.html)

# COMMAND ----------

algo = xgb.XGBRegressor

params = {
  'pca_params': {'n_components': 12},
  'algo_params': {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100, 'objective': 'reg:squarederror'}
}

run_name = "XGBoost"
df_train, df_test, version = prep_data(TABLE_NAME)
run_info, model_stats = run_algo(df_train, df_test, version, FEATURES, algo, params, experiment_id, run_name=run_name, do_shap=True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Now you have tested 2 different models. Check the Experiment and compare the results.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hyperparameter Tuning
# MAGIC User HyperOpt with Spark trials to run distributed hyperparameters tuning across workers in parallel

# COMMAND ----------

spark.conf.set("spark.databricks.mlflow.trackHyperopt.enabled", False)

# COMMAND ----------

from functools import partial
from hyperopt import SparkTrials, hp, fmin, tpe, STATUS_FAIL, STATUS_OK

spark_trials = SparkTrials()
hyperopt_algo = tpe.suggest

n_components_range = np.arange(4, 12, 1, dtype=int)
max_depth_range = np.arange(1, 4, 1, dtype=int)
learning_rate_range = np.arange(0.01, 0.15, 0.01)
n_estimators_range = np.arange(100, 500, 1, dtype=int)

params = {
  'pca_params': {
    'n_components': hp.choice('n_components', n_components_range)
  },
  'algo_params': {
    'max_depth': hp.choice('max_depth', max_depth_range), 
    'learning_rate': hp.choice('learning_rate', learning_rate_range), 
    'n_estimators': hp.choice('n_estimators', n_estimators_range), 
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'early_stopping_rounds': 100,
    'verbose': False
  }
}

def rmse(y, pred): 
  return np.sqrt(mean_squared_error(y, pred))

def fn(params, df_train, df_test, version, features, algo, experiment_id, run_name):
  run_info, model_stats = run_algo(df_train, df_test, version, features, algo, params, experiment_id=experiment_id, run_name=run_name, do_shap=True, nested=True)
  
  loss = rmse(model_stats['df']['PRICE'].to_numpy(), model_stats['y_pred'])
  return {'loss': loss, 'status': STATUS_OK}

run_name = "HyperOpt - XGB"

algo = xgb.XGBRegressor
df_train, df_test, version = prep_data(TABLE_NAME)
fmin_objective = partial(fn, df_train=df_train, df_test=df_test, version=version, features=FEATURES, algo=algo, experiment_id=experiment_id, run_name=run_name)

best_param = fmin(fn=fmin_objective, space=params, algo=hyperopt_algo, max_evals=8, trials=spark_trials) 


# COMMAND ----------

print("best hyperparameters:")
print(f"learning_rate: {learning_rate_range[best_param['learning_rate']]}")
print(f"max_depth: {max_depth_range[best_param['max_depth']]}")
print(f"n_estimators: {n_estimators_range[best_param['n_estimators']]}")
print(f"n_components: {n_components_range[best_param['n_components']]}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Now you have found the best parameters for your XGBoost model, we can deploy it using the next [notebook]($./03_deployment).