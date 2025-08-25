#!/usr/bin/env python

# Import library
import pandas as pd
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *
import sys
from pmdarima import auto_arima
import statsmodels.api as sm


# Import the csv file and explore it
city_pd = pd.read_csv(sys.argv[1])
# Convert ds to datetime
city_pd['date'] = pd.to_datetime(city_pd['date'])
#city_pd['AvgTemperature'] = city_pd['AvgTemperature']
# Display info
city_pd.info()

# Unique cities
print(city_pd[['City']].nunique())

df = city_pd[['date', 'City', 'AvgTemperature']]
df.set_index(['City', 'date'], inplace=True)
df.head()

# Aggregate data by month
df = df.groupby([pd.Grouper(level='City'),
                 pd.Grouper(level='date', freq='M')]).mean()

df.reset_index(inplace=True)

spark = SparkSession\
    .builder\
    .master("yarn")\
    .appName("CityTempsSarimax")\
    .getOrCreate()

# Use the Cloud Storage bucket for temporary BigQuery export data used
# by the connector.
bucket = "preds_bucket_44853"
spark.conf.set('temporaryGcsBucket', bucket)

# Read the csv file
city_df = spark.createDataFrame(df)
# Display the schema
city_df.printSchema()

# Partition the data
city_df.createOrReplaceTempView("cities")
sql = "select * from cities"
cities_part = (spark.sql(sql).repartition(6, ['City'])).cache()
cities_part.explain()

# Define a schema
schema = StructType([
                     StructField('City', StringType()),
                     StructField('date', TimestampType()),
                     StructField('AvgTemperature', DoubleType()),
                     StructField('PredictedTemps', DoubleType()),
])


# define the Pandas UDF
@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def apply_model(city_pd):
    city_pd.set_index('date', inplace=True)
    stepwise_fit = auto_arima(city_pd['AvgTemperature'], trace=True,
                              suppress_warnings=True)

    print(city_pd.shape)
    train = city_pd.iloc[:-24]
    test = city_pd.iloc[-24:]
    first_date = test.index[0]
    print(train.shape, test.shape)

    model = sm.tsa.statespace.SARIMAX(train['AvgTemperature'],
                                      order=stepwise_fit.order,
                                      seasonal_order=(1, 1, 1, 12))
    model = model.fit()
    model.summary()

    start = len(train)
    # Predict 3 years into future from last available data
    end = (len(train)+len(test)-1) + 3*12
    pred = model.predict(start=start, end=end).rename('ARIMA Predictions')

    pred_series = pred
    newtest = test
    ix = pd.date_range(start=first_date, periods=60, freq='M')
    newtest = newtest.reindex(ix)
    newtest['PredictedTemps'] = pred_series

    # Create a data frame that contains store, item, y, and yhat
    f_pd = newtest.reset_index()
    f_pd = f_pd.rename(columns={'index': 'date'})
    result_pd = f_pd
    # fill store and item
    result_pd['City'] = city_pd['City'].iloc[0]
    final_result = result_pd

    return final_result


# Apply the function to all store-items
results = cities_part.groupby(['City']).apply(apply_model)

# Saving the data to BigQuery
results.write.format('bigquery') \
  .option('table', 'temp_predictions_dataset.sarimax_CityState_monthly') \
  .save()


