# Databricks notebook source
# MAGIC %pip install shap

# COMMAND ----------

import numpy as np
import shap
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt

from plotly.graph_objs import *
from plotly.offline import plot

import warnings
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA

import xgboost as xgb

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Define Machine Learning model
# MAGIC 
# MAGIC We are mimicing Sklearn API and have a Machine Learning model that does: 
# MAGIC - Dimentionality Reduction with PCA
# MAGIC - Training with arbitrary ML algo and set of hyper-parameters

# COMMAND ----------

# Define a class that somewhat mimics sklearn API 
# This class might have some steps to do preprocessing with data, such as: 
# - scaling
# - lookups
# - feature engineering

class BostonModel(object):
  def __init__(self, algo, pca_params={}, algo_params={}):
    self._algo = algo
    self._algo_params = algo_params
    self._pca_params = pca_params 
    self._label_col = 'PRICE'
  
  def train(self, df):
    X = df.drop(self._label_col, axis=1)
    self.columns_ = X.columns

    y = df[self._label_col].to_numpy()
    
    self._pca = PCA(**self._pca_params).fit(X)
    X = self._pca.transform(X)
    
    self._model = self._algo(**self._algo_params).fit(X, y)
    return self

  def predict(self, X):
    X = self._pca.transform(X)
    preds = self._model.predict(X)
    return preds

  def predict_pd(self, df):
    X = df[self.columns_]
    preds = self.predict(X)
    return preds

  
def evaluate(y_true, y_pred, prefix='test'):
  out = {
    '{}_mse'.format(prefix): mean_squared_error(y_true, y_pred),
    '{}_mae'.format(prefix): mean_absolute_error(y_true, y_pred),
    '{}_r2'.format(prefix): r2_score(y_true, y_pred)
  }
  
  return out

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run & Evaluate cycle
# MAGIC - Train Machine Learning model
# MAGIC - Create predictions for test & train datasets 
# MAGIC - Evaluate performance
# MAGIC - Log run attributes to ML-Flow 

# COMMAND ----------

global scatter_test
global hist_test
global shap_chart

get_algo_name = lambda algo: str(algo).split("'")[1].split('.')[-1]

# run algorithm wrapper
# this will do end-to-end evaluation of our model
# and log certain attributes to MlFlow

def prep_data(table_name):
  
  version = spark \
    .sql(f'SELECT max(version) AS current_version FROM (DESCRIBE history {table_name})') \
    .collect()[0]['current_version']
  
  df = spark.sql(f'SELECT * FROM {table_name}').toPandas()
  
  df_train, df_test = train_test_split(df, random_state=42)
  
  return df_train, df_test, version


def run_algo(df_train, df_test, version, features, algo, params, experiment_id, run_name=None, do_shap=False, verbose=True, nested=False):
  _print = print if verbose else lambda x: x
  
  pca_params = params['pca_params']
  algo_params = params['algo_params']
  
  model = BostonModel(
      algo=algo,
      pca_params=pca_params,
      algo_params=algo_params
   ).train(df_train)

  _print('done: trained model')
  
  y_train_pred = model.predict_pd(df_train)
  _print('done: predicting train'.format(y_train_pred))
  
  y_test_pred = model.predict_pd(df_test)
  _print('done: predicting test'.format(y_test_pred))

  metrics_train = evaluate(df_train['PRICE'], y_train_pred, prefix='train')
  _print('metrics train: {}'.format(metrics_train))
  
  metrics_test = evaluate(df_test['PRICE'], y_test_pred, prefix='test')
  _print('metrics test: {}'.format(metrics_test))
  
  with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=nested) as run:
    mlflow.log_params(pca_params)
    mlflow.log_params(algo_params)
    mlflow.log_params({'algo_name': get_algo_name(algo)})
    mlflow.log_params({'table_version': version})
    
    mlflow.log_metrics(metrics_train)
    mlflow.log_metrics(metrics_test)
    
    mlflow.set_tag('features', features)
    mlflow.set_tag('run_id', run.info.run_uuid)
    
    mlflow.sklearn.log_model(model, 'model')
  
    scatter_test = get_scatter(y_test_pred, df_test['PRICE'], 'test')
    mlflow.log_artifact('scatter_test.png', 'charts')

    hist_test = get_hist(y_test_pred, df_test['PRICE'], 'test')
    mlflow.log_artifact('hist_test.png', 'charts')
    
    if do_shap:
      shap_mean = get_shap_mean_values(model, df_test, FEATURES)
      plot(get_mean_shap_feaute_importance(shap_mean, FEATURES), filename='shap.html')
      mlflow.log_artifact('shap.html', 'charts')
    
    _print('done run: id={}'.format(run.info.run_uuid))
  
  plt.close('all')
  return run.info, {'y_pred': y_test_pred, 'df': df_test, 'model': model}

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Visualisation

# COMMAND ----------

# Utility visualisation functions to get:
# - scatterplot of predictions vs true labels
# - histogram of predictions vs histogram of true labels

def get_scatter(y_pred, y_true, dset='test'):
  f, ax = plt.subplots()
  
  ax.scatter(y_pred, y_true)
  ax.set_title('y_pred vs y_true scatter: {}'.format(dset))
  ax.set_xlabel('y_pred')
  ax.set_ylabel('y_true')
  
  f.savefig('scatter_{}.png'.format(dset))
  return f
  
  
def get_hist(y_pred, y_true, dset='test'):
  f, (ax_pred, ax_true) = plt.subplots(1, 2, figsize=(8, 4))
  f.suptitle('y_pred vs y_true hist: {}'.format(dset))

  ax_pred.hist(y_pred)
  ax_pred.set_title('y_pred')
  
  ax_true.hist(y_true)
  ax_true.set_title('y_true')
  
  f.savefig('hist_{}.png'.format(dset))
  return f

# COMMAND ----------

def get_mean_shap_feaute_importance(mean_shap_values, features):
  argsort_shap_vals = np.argsort(mean_shap_values)

  trace1 = {
    "uid": "b35d8a7c-bec0-4b9e-be32-563e8da2c6f8", 
    "fill": "toself", 
    "mode": "markers+lines", 
    "name": "Shap mean feature importance", 
    "r": mean_shap_values[argsort_shap_vals], 
    "type": "scatterpolar", 
    "marker": {"size": 5}, 
    "theta": [features[i] for i in argsort_shap_vals]
  }
  
  data = Data([trace1])
  
  layout = {
    "polar": {
      "bgcolor": "rgb(243,243,243)", 
      "radialaxis": {
        "side": "counterclockwise", 
        "visible": True, 
        "showline": True, 
        "gridcolor": "white", 
        "gridwidth": 2, 
        "linewidth": 2, 
        "tickwidth": 2
      }, 
      "angularaxis": {
        "layer": "below traces", 
        "tickfont": {"size": 10}
      }
    }, 
    "title": {"text": "Feature Importance"}, 
    "height": 700,
    "paper_bgcolor": "rgb(243,243,243)"
  }
  fig = Figure(data=data, layout=layout)
  return fig

# COMMAND ----------

def get_shap_mean_values(model, df, features):
  explainer = shap.KernelExplainer(model=model.predict, data=shap.kmeans(df[features], 10))
  shap_values = explainer.shap_values(df[FEATURES].sample(10), l1_reg='aic', silent=True)
  
  mean_shap_values = shap_values.mean(axis=0)
  return mean_shap_values

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Scoring

# COMMAND ----------

def score_dataset(df, model):
  ts = datetime.now()
  pred = df.loc[:,['PRICE']]
  pred['PREDICTION'] = model.predict_pd(df)
  pred['TS'] = ts
  
  return pred