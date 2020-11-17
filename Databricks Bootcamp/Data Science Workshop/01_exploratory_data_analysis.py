# Databricks notebook source
# MAGIC %run ./00_setup

# COMMAND ----------

# MAGIC %md ## Load & Examine the data
# MAGIC What do we want to do here: 
# MAGIC - Load data from Delta
# MAGIC - Register our Spark Dataframe as a temporary view to run SQL.
# MAGIC - Run some visualisations on it.

# COMMAND ----------

df = spark.read.format('delta').load('/demo/ML/boston_house_price')
df.createOrReplaceTempView('boston_house_price')

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Data Dictionary
# MAGIC 
# MAGIC We will be predicting the price of a property using different attributes as we can see from the table below
# MAGIC 
# MAGIC |Column|Description|
# MAGIC |-----|-----|
# MAGIC |BRIM|per capita crime rate by town.|
# MAGIC |ZN|proportion of residential land zoned for lots over 25,000 sq.ft.|
# MAGIC |INDUS|proportion of non-retail business acres per town.|
# MAGIC |CHAS|Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).|
# MAGIC |NOX|nitrogen oxides concentration (parts per 10 million).|
# MAGIC |RM|average number of rooms per dwelling.|
# MAGIC |AGE|proportion of owner-occupied units built prior to 1940.|
# MAGIC |DIS|weighted mean of distances to five Boston employment centres.|
# MAGIC |RAD|index of accessibility to radial highways.|
# MAGIC |TAX|full-value property-tax rate per $10,000.|
# MAGIC |PTRATIO|pupil-teacher ratio by town.|
# MAGIC |B|1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.|
# MAGIC |LSTAT|lower status of the population (percent).|

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from boston_house_price

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inspect the Dataset
# MAGIC We can use both native and external visulisation libraries to help us get a better understanding of the data

# COMMAND ----------

# DBTITLE 1,2 Way Scatter Plot
# MAGIC %sql
# MAGIC select * from boston_house_price

# COMMAND ----------

# DBTITLE 1,Histogram
df = df.toPandas()

# display works with Pandas & Spark alike
display(df)

# COMMAND ----------

# DBTITLE 1,Correlation Matrix (using seaborn)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(7, 7))
display(sns.heatmap(df.corr(), cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 10}, cmap='Greens'))

# COMMAND ----------

# DBTITLE 1,Scaled Scatter Plot (using Plotly)
import pandas as pd
import plotly.express as px

df_scaled = df / df.max()

df_scaled_melt = pd.melt(
  df_scaled,
  id_vars=['PRICE'],
  value_vars=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],
  var_name=['FEATURE'],
  value_name='NORM_VALUE'
)

fig = px.scatter(
  df_scaled_melt, 
  x='PRICE', 
  y='NORM_VALUE', 
  color='FEATURE', 
  hover_data=['FEATURE'], 
  opacity=0.3, 
  trendline='lowess', 
  size_max=10
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Now we have some basic understanding of the data, let's build some ML models using the next [notebook]($./02_machine_learning).