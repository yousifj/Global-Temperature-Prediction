#!/usr/bin/env python

# Import library
import pandas as pd
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *
from prophet import Prophet
import sys

# Import the csv file
city_pd = pd.read_csv(sys.argv[1])
# Convert ds to datetime
city_pd['ds'] = pd.to_datetime(city_pd['date'])
city_pd['y'] = city_pd['AvgTemperature']

# Unique cities
print(city_pd[['City']].nunique())

spark = SparkSession\
    .builder\
    .master("yarn")\
    .appName("CityTemps")\
    .getOrCreate()

# Use the Cloud Storage bucket for temporary BigQuery export data used
# by the connector.
bucket = "preds_bucket_44853"
spark.conf.set('temporaryGcsBucket', bucket)

# Create Spark dataframe based on original Pandas dataframe
city_df = spark.createDataFrame(city_pd)
# Display the schema
city_df.printSchema()

# Partition the data
city_df.createOrReplaceTempView("cities")
sql = "select * from cities"
cities_part = (spark.sql(sql)\
    .repartition(6,
   #.repartition(spark.sparkContext.defaultParallelism,
   ['City'])).cache()
cities_part.explain()

# Define a schema
schema = StructType([
                     StructField('City', StringType()),
                     StructField('ds', TimestampType()),
                     StructField('y', FloatType()),
                     StructField('yhat', DoubleType()),
                     StructField('yhat_upper', DoubleType()),
                     StructField('yhat_lower', DoubleType()),
])


# define the Pandas UDF
@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def apply_model(city_pd):
    # instantiate the model and set parameters
    model = Prophet()
    # fit the model to historical data
    model.fit(city_pd)
    # Predict 3 years into future from latest available data
    future = model.make_future_dataframe(periods=1095)
    # Out of sample prediction
    future = model.predict(future)
    # Create a data frame that contains store, item, y, and yhat
    f_pd = future[['ds', 'yhat', 'yhat_upper', 'yhat_lower']]
    ct_pd = city_pd[['ds', 'City', 'y']]
    result_pd = f_pd.join(ct_pd.set_index('ds'), on='ds', how='left')
    # fill store and item
    city_name = city_pd['City'].iloc[0]
    result_pd['City'] = city_pd['City'].iloc[0]
    final_result = result_pd[['City', 'ds', 'y', 'yhat',
                              'yhat_upper', 'yhat_lower']]
    return final_result


# Apply the function to all store-items
results = cities_part.groupby(['City']).apply(apply_model)

# Saving the data to BigQuery
results.write.format('bigquery') \
  .option('table', 'temp_predictions_dataset.Prophet_CityState') \
  .save()


# Partially based on https://medium.com/@y.s.yoon/scalable-time-series-forecasting-in-spark-prophet-cnn-lstm-and-sarima-a5306153711e