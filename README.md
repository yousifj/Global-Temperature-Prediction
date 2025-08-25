Files included in this repo

		Project Report.pdf : Final report in IEEE Latex Format. 
		
		prophet_pyspark_bq.py : Code to run Pyspark job for daily city 
		temperature predictions using Prophet model.

		sarimax_pyspark_bq_monthly.py : Code to run Pyspark job for monthly city 
		temperature predictions using SARIMAX model.

		sarimax_pyspark_bq_weekly.py : Code to run Pyspark job for weekly city 
		temperature predictions using SARIMAX model.
		
		index.html : Full D3 code for the webpage
				   : To run this file set up Apache HTTP Server and move
				   all of the .csv files to the same directory. 
				   Then open local host http://127.0.0.1/ in your browser.
		
		
		ARIMA.ipynb : jupyter-notebook used to test ARIMA(for testing only).
		
		LSTM.ipynb  : jupyter-notebook used to test LSTM(for testing only).
		
		Max Temperatures.ipynb: jupyter-notebook used to get Max Temperatures for each city
					with the date (for analyzing and understanding the data).
		
		single-city-prophet-mae.ipynb : Jupyter notebook written on Kaggle 
		exploring Prophet model functionality when making predictions for a single city (San Diego). 
	
		project_eda.ipynb : Jupyter notebook written to help us clean the data 
		and create the new CSV file with our cleaned data. 

		city_temperature.csv : Raw Data from Kaggle

		city_temperature_clean.csv : Cleaned Data for Processing
		
		Prophet Data.csv: contains data predicted using prophet
					y is original value, y_hat is the predicted and y_lower, y_upper 
					are the confidence interval.
					
		SARIMAX 2 years data.csv: contains data predicted using SARIMAX Weekly
					AvgTemperature is original value, and PredictedTemps it the predicted.
					
		SARIMAX 2 years data Monthly.csv: contains data predicted using SARIMAX Monthly
					AvgTemperature is original value, and PredictedTemps it the predicted.

GitHub note: Make sure to extract all ZIP files for the data. They were compressed to bypass GitHub’s 100 MB file size limit.
