Melbourne CBD Urban Pedestrian Climate Impact Prediction

Client: City of Melbourne

## Table of Contents

- [0. Introduction](#0-introduction)
- [1. Importing The Libraries](#1-importing-the-libraries)
- [2. Importing The Datasets](#2-importing-the-datasets)
  - [2.1 Pedestrian Counting System - Sensor Locations](#21-pedestrian-counting-system---sensor-locations)
  - [2.2 Pedestrian Counting System (counts per hour)](#22-pedestrian-counting-system-counts-per-hour)
  - [2.3 Microclimate sensors data](#23-microclimate-sensors-data)
- [3. Initial Inspection Of The Datasets](#3-initial-inspection-of-the-datasets)
  - [3.1 Checking Number Of Rows/Columns](#31-checking-number-of-rowscolumns)
  - [3.2 Checking The Features](#32-checking-the-features)
  - [3.3 Checking The Datatypes](#33-checking-the-datatypes)
  - [3.4 Checking The Dataset Information](#34-checking-the-dataset-information)
  - [3.5 Checking For Missing Values](#35-checking-for-missing-values)
  - [3.6 Checking Summary Statistics](#36-checking-summary-statistics)
  - [3.7 Checking The Date Format](#37-checking-the-date-format)
  - [3.8 Checking The ID Columns](#38-checking-the-id-columns)
- [4. Data Cleaning](#4-data-cleaning)
  - [4.1 Removing Irrelevant Columns](#41-removing-irrelevant-columns)
  - [4.2 Removing Missing Value](#42-removing-missing-value)
  - [4.3 Datetime Formatting](#43-datetime-formatting)
  - [4.4 Aggregating Values For Hourly Format](#44-aggregating-values-for-hourly-format)
  - [4.5 Merging The Datasets](#45-merging-the-datasets)
- [5. Data Validation](#5-data-validation)
  - [5.1 Validating Time Series Dataset](#51-validating-time-series-dataset)
  - [5.2 Fixing Missing Timestamps](#52-fixing-missing-timestamps)
- [6. Exploratory Data Analysis](#6-exploratory-data-analysis)
  - [6.1 Pedestrian Count Over Time (Hourly)](#61-pedestrian-count-over-time-hourly)
  - [6.2 Average Pedestrian Count By Day Of Week](#62-average-pedestrian-count-by-day-of-week)
  - [6.3 Distributions Of Variables](#63-distributions-of-variables)
  - [6.4 Correlation Matrix](#64-correlation-matrix)
- [7. Time Series Analysis](#7-time-series-analysis)
  - [7.1 Pedestrian Count Over Time (24-Hour Rolling Mean)](#71-pedestrian-count-over-time-24-hour-rolling-mean)
  - [7.2 Seasonal Pattern (24-Hour Cycle)](#72-seasonal-pattern-24-hour-cycle)
  - [7.3 Autocorrelation Of Pedestrian Count](#73-autocorrelation-of-pedestrian-count)
  - [7.4 Augmented Dickey-Fuller Test](#74-augmented-dickey-fuller-test)
- [8. Feature Engineering](#8-feature-engineering)
  - [8.1 Creating Time Features](#81-creating-time-features)
  - [8.2 Creating Cyclical Time Features](#82-creating-cyclical-time-features)
  - [8.3 Creating Cyclical Wind Direction Features](#83-creating-cyclical-wind-direction-features)
  - [8.4 Creating Lag Features](#84-creating-lag-features)
  - [8.5 Creating Rolling Features](#85-creating-rolling-features)
  - [8.6 Removing NA Rows](#86-removing-na-rows)
  - [8.7 Removing Unnecessary Features](#87-removing-unnecessary-features)
- [9. Preparing Train/Val/Test Splits](#9-preparing-trainvaltest-splits)
  - [9.1 Splitting The Data By Time](#91-splitting-the-data-by-time)
  - [9.2 Separating Features And Target](#92-separating-features-and-target)
  - [9.3 Scaling The Features](#93-scaling-the-features)
  - [9.4 Creating LSTM Sequences](#94-creating-lstm-sequences)
- [10. Baseline LSTM Model](#10-baseline-lstm-model)
  - [10.1 Model Building](#101-model-building)
  - [10.2 Baseline Loss Plot](#102-baseline-loss-plot)
- [11. Hyperparameter Tuning LSTM Model](#11-hyperparameter-tuning-lstm-model)
  - [11.1 Model Building With Hyperparameter Comparisons](#111-model-building-with-hyperparameter-comparisons)
- [12. Evaluating Model Performance](#12-evaluating-model-performance)
  - [12.1 Baseline Vs Best Hyperparameter Tuning Model](#121-baseline-vs-best-hyperparameter-tuning-model)
  - [12.2 Actual Vs Predicted Values Plot Comparisons](#122-actual-vs-predicted-values-plot-comparisons)
- [13. Using The Best Model](#13-using-the-best-model)
  - [13.1 Predicting With The Best Model](#131-predicting-with-the-best-model)
  - [13.2 Preparing The Latest Input Window](#132-preparing-the-latest-input-window)
  - [13.3 Making The Prediction](#133-making-the-prediction)
  - [13.4 Bar Chart Of Actual Vs Predicted](#134-bar-chart-of-actual-vs-predicted)
- [14. References](#14-references)

---

# Urban Pedestrian Climate Impact Prediction

**Authored by:** Maverick Nguyen

**Duration:** 40 mins 

**Pre-requisite Skills:** Python, Data Cleaning, Data Visualisation, Time-Series Analysis, Feature Engineering, Optimisation Methods, Deep Learning

**User Story**

As a local living near Melbourne CBD, Maverick relies on active travel, like walking, and public transport, like trams, to get around to different places he wants to go. One morning in January, Maverick prepared to travel to his workplace, expecting to get to work before 9:00 AM, but there was a sudden heatwave, causing the tram that he usually catches to be unable to follow its designated schedule and creating a delay in his schedule. Although there was a replacement bus for this emergency, only a limited number of people could board this vehicle, which further delayed his schedule. Because of this sudden extreme weather, travel conditions become less reliable and difficult to predict.

Maverick wants to have access to a system that could predict how climate conditions over time can affect urban pedestrian movement. So that he could better plan his trip, allowing him to leave earlier in anticipation of sudden extreme weather change during a particular timeframe, or choose a different mode of transport, like an Uber. This allows more support in making informed decisions when travelling during extreme weather events.

**What this use case will teach you**

At the end of this use case you will:
- Learn how to source and combine multiple public datasets.  
- Understand how to clean and align time-series data at an hourly level for modelling. 
- Explore how climate variables, such as temperature, humidity, pressure, and wind, relate to pedestrian counts.
- Apply feature engineering techniques to create meaningful predictors from weather and mobility time-series data. 
- Build a deep learning forecasting model to predict pedestrian demand. 
- Perform model optimisation like hyperparameter tuning to improve forecasting performance. 
- Evaluate model performance and interpret results for climate adaptation planning.

## 0. Introduction

Urban systems are often affected by changing climate conditions, but these effects are not always easy to capture with simple forecasting methods. One clear example is pedestrian movement, where changing weather conditions can affect how many people move through the city over time.

This use case focuses on predicting pedestrian activity in the City of Melbourne using hourly climate observations, which keeps the project closely aligned with the goal of modelling how climate factors influence an urban system.

In this use case, pedestrian counts are aggregated into hourly city-level totals and merged with hourly microclimate observations for Melbourne. A deep learning model can then be trained to predict pedestrian demand based on time, recent demand history, and recent climate conditions.

The datasets used in this project are the "Pedestrian Counting System (counts per hour)", the "Pedestrian Counting System - Sensor Locations" dataset for supporting location metadata, and the "Microclimate Sensor Readings" dataset from the City of Melbourne website.

## 1. Importing The Libraries

This section is to show what libraries were used for this use case, with each imported library supporting a specific part of the pipeline. These libraries are necessary for doing data handling, time-series analysis, visualisation, feature engineering, optimisation, and deep learning (1, 2, 3, 4, 5). These were added at the beginning to ensure the workflow is organised (6). A random seed was set with a student ID to allow reproducibility of the outputs in this notebook (7).

## 2. Importing The Datasets

This section is necessary for importing multiple public datasets from the City of Melbourne, before any cleaning, merging and modelling in later stages can happen. These datasets will be accessed through the City of Melbourne Open Data API v2.1, to allow the notebook to be directly used upon download (8).

By using a shared BASE_URL and a dictionary of dataset identifiers, this allows for removing and adding datasets more easily (8, 9, 10, 11). The get_csv_url() function is especially useful because it standardises the dataset access method and allows the same logic to be reused across all three datasets (8). The ROW_LIMIT parameter was added to allow experimentation on smaller samples before scaling to the full dataset (8). Lastly, the datasets are accessed through the Melbourne Open Data API v2.1, and no visible API key is exposed in the code (8).

### 2.1 Pedestrian Counting System - Sensor Locations

**Justification**

This dataset was included to provide contextual information about the physical pedestrian counting network, even though it wasn't used in the later steps for modelling. The sensor metadata helps explain where the mobility data originates from and what the coverage of the system looks like (9).

**Output Explanation**

* `location_id`: unique identifier for each pedestrian counting location.
* `sensor_description`: human-readable description of the site, like street names.
* `sensor_name`: short internal sensor code.
* `installation_date`: date the sensor was installed.
* `note`: extra comments or metadata about the sensor.
* `location_type`: type of location.
* `status`: operational status of the sensor.
* `direction_1` and `direction_2`: the two movement directions captured by the counter.
* `latitude` and `longitude`: geographic coordinates of the sensor.
* `location`: combined coordinate string.

### 2.2 Pedestrian Counting System (counts per hour)

**Justification**

This is the target dataset for the use case because the final prediction task is to forecast pedestrian demand. The pedestrian dataset contains the observed mobility outcome that the model aims to learn (10).

**Output Explanation**

* `id`: record identifier.
* `location_id`: identifier linking the observation to a specific sensor location.
* `sensing_date`: date of observation.
* `hourday`: hour of day from 0 to 23.
* `direction_1` and `direction_2`: directional pedestrian counts.
* `pedestriancount`: total pedestrian count for that record.
* `sensor_name`: short sensor code.
* `location`: coordinate string for the sensor. 

### 2.3 Microclimate sensors data

**Justification**

This dataset provides the microclimate data that the use case requires to understand how climate conditions affect urban pedestrian movement, more specifically, the input features. It basically provides the explanatory environmental variables to connect weather conditions to mobility demand (11).

**Output Explanation**

* `device_id`: identifier for each microclimate device.
* `received_at`: timestamp when the reading was recorded.
* `sensorlocation`: descriptive location of the microclimate sensor.
* `latlong`: coordinate string.
* `minimumwinddirection`, `averagewinddirection`, `maximumwinddirection`: wind direction measurements.
* `minimumwindspeed`, `averagewindspeed`, `gustwindspeed`: wind speed measurements.
* `airtemperature`: air temperature reading.
* `relativehumidity`: humidity level.
* `atmosphericpressure`: atmospheric pressure.
* `pm25` and `pm10`: particulate matter measurements.
* `noise`: noise level. 

## 3. Initial Inspection Of The Datasets

This section is necessary to understand the size, structure, completeness and formatting of public datasets since they often differ. Public datasets can have different structures, missing values, data types, date formats, and identifier systems, so checking them early helps identify potential issues in the workflow (12). Before trying to clean and merge the datasets, understanding what each of the datasets contains and how they can be combined is important (12).

### 3.1 Checking Number Of Rows/Columns

**Justification**

Checking the shape is important because it shows the scale and complexity of each dataset. This helps determine memory demands, cleaning strategy, and whether the data volume is sufficient for later modelling (13).

**Output Explanation**

From these results, the pedestrian and microclimate datasets are large enough for later time-series modelling. The sensor locations dataset is much smaller because it only contains metadata about the sensor network, rather than repeated hourly observations. The volume for each task varies, but from the inspection of the shapes, there appears to be sufficient volume for the task of predicting the pedestrian count.

### 3.2 Checking The Features

**Justification**

This step is to verify what information is actually available in each dataset, and whether there are meaningful fields for later joining, cleaning, and modelling. Knowing the columns early prevents accidentally removing important variables or keeping irrelevant variables (14).

**Output Explanation**

From the output, the column names confirm that location_id is shared between the sensor metadata and pedestrian counts datasets. Also, the pedestrian and microclimate datasets both contain time information, which is essential because the final merge is ultimately done at the hourly level. The microclimate dataset clearly offers a diverse range of explanatory variables, which is useful for modelling. And, some columns that are likely less useful for the final model need to be removed, such as descriptive notes, raw coordinate strings, and duplicate directional fields.

### 3.3 Checking The Datatypes

**Justification**

Datatype checking is essential because many later operations depend on correct types to proceed. Steps like date parsing, numeric aggregation, interpolation, rolling windows, and model preparation can all fail or behave incorrectly if types are wrong (15).

**Output Explanation**

From the outputs, the pedestrian sensing_date, sensor installation_date, and microclimate received_at fields are all initially stored as objects, so they are not yet ready for time-series operations, which will need to be dealt with. And the numeric climate variables are already in float64, and pedestrian counts are in int64, which is appropriate for aggregation and modelling, so no need to change that.

### 3.4 Checking The Dataset Information

**Justification**

Using .info() gives a more complete structural summary of the datasets, which shows non-null counts, memory usage, and dtype balance. But this step will focus more on the memory that will be used up in the RAM, to decide what sample size would be ideal for experimentation before scaling to the full dataset size, or opt for a platform like Google Colab to handle more heavy usage (16).

**Output Explanation**

From the outputs, the pedestrian dataset takes up the most memory, followed by the microclimate dataset, with the sensor locations dataset being extremely low. Which is acceptable to run locally for modelling.

### 3.5 Checking For Missing Values

**Justification**

Checking for missing values is important, since it affects the cleaning strategy, feature selection, and merge quality. This is especially important because missing sensor readings can break hourly continuity for the modelling later (17).

**Output Explanation**

From the outputs, the pedestrian counts dataset has no missing values at all. The microclimate dataset has many missing values in all the variables besides device_id and received_at. And the sensor locations dataset has some missing values, mainly in note, direction_1, and direction_2, which are not necessary for city-level forecasting.

### 3.6 Checking Summary Statistics

**Justification**

Checking the summary statistics is an easy way to provide some early insights into the datasets, like central tendency and spread (18).

**Output Explanation**

From the sensor coordinates with the mean latitude and longitude, the Melbourne CBD can be inferred to be the main area of focus for the datasets. The pedestrian counts are strongly right-skewed, with the median count being a fair bit lower than the mean, and the maximum being really high, which suggests that some hours and sites are much busier than others. The average microclimate temperature is about 16°C, and the average relative humidity is about 66%, which looks plausible for Melbourne across a long time range.

### 3.7 Checking The Date Format

**Justification**

Checking the date format is important because a time-based merge depends on all datasets sharing a compatible datetime structure, and public datasets often use different date formats and timezone conventions. This is important because any mismatch in date or time formatting can prevent the datasets from merging correctly later (19).

**Output Explanation**

The pedestrian dataset stores dates and hours separately. Whereas the microclimate dataset stores the full timestamps with date and time with UTC offsets. And the sensor installation date is a simple date string and is mainly historical metadata rather than a modelling field. This makes it clear that datetime standardisation is a required step before any merge can occur.

### 3.8 Checking The ID Columns

**Justification**

This was an additional step for checking the unique identifiers to see if the datasets have any chance of being joined directly by ID or whether another strategy, like a datetime merge, is best. And since the project uses multiple public datasets, checking the unique IDs also helps confirm whether the pedestrian sensors and microclimate sensors use the same location system or separate systems (20).

**Output Explanation**

From the outputs, the sensor locations dataset contains 137 unique location_id values, while the pedestrian counts dataset contains 100 unique location_id values. Whereas the microclimate dataset contains 12 unique device_id values, which is a completely different identifier system. This means the microclimate data cannot be joined to pedestrian counts by location ID, so a time-based merge is the best integration method available.

## 4. Data Cleaning

This is the section for data cleaning, since raw data are rarely ready to be used as it is, so cleaning processes are necessary to address duplicates, missing values, inconsistencies, syntax errors, irrelevant data and structural errors (21, 22). And since this data pipeline uses the API for dataset access, this means that the datasets are being updated in real-time, so ensuring any errors get addressed in the pipeline ensures the data remains accurate, secure and accessible at every stage of its lifecycle (21). And that the prediction will also be accurate (22).

### 4.1 Removing Irrelevant Columns

**Justification**

Selecting relevant variables and removing the irrelevant ones are necessary because not every feature is useful for the prediction modelling. Keeping unnecessary columns can make the workflow harder to manage, increase memory usage, and create confusion in later steps (23, 24).

**Output Explanation**

In this step, the sensor dataset is reduced to six useful metadata columns, even though it wasn't used for modelling purposes. The pedestrian dataset is reduced to the four fields needed to construct hourly counts. The microclimate dataset is reduced to key climate, air quality, and noise variables. The descriptive or duplicate variables were omitted.

### 4.2 Removing Missing Value

**Justification**

This step involves dealing with missing values by removing the rows they're in. This is because missing values in explanatory variables can cause problems when doing aggregation, interpolation, and modelling later, leading to bias results (25). This is especially important for the microclimate dataset, since missing climate readings could affect the quality of the explanatory variables.

**Output Explanation**

After doing the column selection in the previous step, the sensor and pedestrian tables are now fully complete without missing values. Whereas the microclimate table still has many missing values, especially in gustwindspeed, pm25, pm10, and noise. 

By using dropna(), the microclimate dataset lost a number of data points with missing values, but still retains a sizable portion of data points. The decision for completeness in the data points was preferred to simplify the merging and feature engineering steps later, and because there were enough data points for it not to matter much.

### 4.3 Datetime Formatting

**Justification**

Ensuring the datetime formatting matches between the different datasets is important because the modelling is hourly, and the datasets need to be able to merge based on a common point for perform chronological analysis across different data sources (26, 27). Skipping this step would mean that the datasets cannot be merged into one table.

**Output Explanation**

The pedestrian dataset is converted from separate sensing_date and hourday fields into a single datetime_hour, such as 2024-12-06 20:00:00. Whereas the microclimate timestamps are converted from UTC into Australia/Melbourne, timezone information is removed, and the values are floored to the nearest hour. By doing this, both datasets now have a datetime variable with the same time formatting. It's also important to note that the time range of the microclimate dataset is narrower than the pedestrian time range, meaning that the overlap period is limited to the microclimate dataset.

### 4.4 Aggregating Values For Hourly Format

**Justification**

This step involves aggregation because the use case models city-level pedestrian demand rather than individual sensor-level behaviour. Since the pedestrian and microclimate datasets both contain multiple records within the same hour. This also ensures that both the pedestrian and the microclimate data share the same hourly rows without duplicates for later merging, reducing the total data volume (28).

**Output Explanation**

The aggregation involves the pedestrian counts being summed across sensors to produce hourly city totals, and the microclimate readings are averaged across devices for each hour. Doing this changes the target variable from pedestrians at one site to overall city pedestrian demand at one hour, along with the climate values at that hour. This also further reduces the row counts due to aggregating to hourly, and the microclimate data is still narrower than the pedestrian dataset.

### 4.5 Merging The Datasets

**Justification**

This step will merge the datasets together, ensuring the target variable, pedestrian count, is connected to the explanatory climate variables. The datetime_hour variable on both the pedestrian dataset and the microclimate dataset was inner-joined to merge into one dataset, meaning only the overlapping rows with the same values were merged (29). Which means every row has both pedestrian and climate information, hence, a unified format ready for analysis (30).

**Output Explanation**

A quick check of the merged dataset shows that the pedestrian counts and climate values aligned in the same hourly observations, which is what the use case is looking for. And there are no missing values, which indicates that previous data cleaning works as intended.

## 5. Data Validation

Doing data validation is important because a merged dataset that was cleaned may still be unsuitable for the task of this use case, possibly due to timestamps being duplicated, out of order, or some rows in the chronological datetime are missing. The time-series model tends to assume a consistent temporal structure with no sudden breaks, so this step in the pipeline checks that everything is complete (31).

### 5.1 Validating Time Series Dataset

**Justification**

Checking for duplicates or missing timestamps, since they can affect the lag features, rolling features and any sequence-based deep learning models that this use case may use. This is to ensure that there is a strictly ordered sequence of evenly spaced time points (32).

**Output Explanation**

From the outputs, there are no duplicates in the datetime_hour values, which means every timestamp is represented once. The dataset is double-checked to ensure that it is sorted in increasing time order. But there appears to be a number of missing hourly timestamps, which means the dataset is not complete yet. It does appear that random points were cut off, and that no large block of time was cut off.

### 5.2 Fixing Missing Timestamps

**Justification**

Ensuring the missing timestamps are filled in is necessary for a complete hourly sequence in the dataset, and missing them can create issues when performing feature engineering later. Missing hours can cause problems when creating lag features, rolling averages, and LSTM input sequences, since these methods rely on consistent time gaps between rows (32, 33).

**Output Explanation**

This step involves reindexing to the full hourly range so that all the missing timestamps are included in the dataset, and then interpolation is performed to fill in the missing values from those created rows. Hence, the merged dataset now has slightly more rows with no missing hourly stamps, ensuring the timeline is continuous with no breaks. A little addition was included to ensure that the pedestrian counts remained as integers, rather than fractions, due to interpolation. 

## 6. Exploratory Data Analysis

The exploratory data analysis section was performed on the merged dataset to observe any interpretable patterns and get a rough understanding of how the dataset looks and feels before building a model. It's important to get a sense of how pedestrian demand changes over time and how it relates to the climate conditions (34).

### 6.1 Pedestrian Count Over Time (Hourly)

**Justification**

Plotting the hourly pedestrian count is for a quick look at the target variable changes over time, to check if the time series data is random or if there are any patterns that may need to be taken into consideration for later steps (35, 36).

**Output Explanation**

From the plot, there appear to be large fluctuations across the full time range with repeated highs and lows. Although it looks random, it does seem to show some sort of pattern, and it's not necessarily random noise, with recurring fluctuations, like the new year of each year seems to always be a high peak, indicating lots of pedestrians travelling in the city, and there also appear to be some sharp dips, which may possibly be public holidays. This does suggest that there may be other possible variables that might have influenced the pedestrian count, but it definitely confirms that pedestrian demand is influenced by recurring temporal and possibly environmental effects.

### 6.2 Average Pedestrian Count By Day Of Week

**Justification**

The day-of-week summary was checked to see if there may also be other variables like work patterns, shopping activities, or weekend behaviours that might have influenced the pedestrian demand. Also checks whether the day of the week should be considered an important time-based feature for later modelling (35, 37).

**Output Explanation**

From the plot, it does seem like the weekdays tend to be relatively high along with Saturdays, whereas Mondays and Sundays tend to have low pedestrian demands. Monday might be low despite being a normal workday for the average Australians might be due to the influence of public holidays, and Sunday being low is due to most people not working on Sunday, since businesses would usually pay a high penalty rate. This further suggests that the CBD pedestrian pattern is linked to weekday economic and commuter activity, not just climate variables.

### 6.3 Distributions Of Variables

**Justification**

Checking for distribution to see whether it's symmetric, skewed, or multi-modal. This is useful because skewed variables, extreme values, or unusual distributions can influence how the model learns from the data (38).

**Output Explanation**

* pedestriancount appears unimodal and right-skewed.
* airtemperature appears unimodal with a slight right skew.
* relativehumidity appears unimodal with a slight left skew.
* atmosphericpressure appears unimodal and slightly left-skewed.
* averagewindspeed appears unimodal and right-skewed.
* gustwindspeed appears unimodal and right-skewed.
* averagewinddirection appears roughly unimodal and mostly symmetric.
* pm25 appears unimodal and right-skewed.
* pm10 appears unimodal and right-skewed, similar to pm25.
* noise appears unimodal and fairly symmetric.

### 6.4 Correlation Matrix

**Justification**

Checking the correlation matrix is a quick way to see which climate variables might have the strongest relationship with pedestrian demand, which does seem like they have some influence on the target variable from the results. This is mainly to show whether the relationship is positive or negative, even though correlation does not prove causation (39).

**Output Explanation**

- noise has the strongest positive correlation with pedestrian count, followed by gustwindspeed, airtemperature, averagewindspeed, and averagewinddirection. This suggests that pedestrian activity tends to increase when these variables increase.
- relativehumidity has the strongest negative correlation with pedestrian count, while pm10, pm25, and atmosphericpressure have weaker negative relationships. This suggests that pedestrian activity tends to decrease when these variables increase.

This suggests that climate variables do play a role in influencing the pedestrian demand, but there are other influences as well as discovered from previous plots. But the goal for this task is to understand how climate variables affect pedestrian demands.

## 7. Time Series Analysis

This section is necessary as exploratory data analysis alone is not enough to check a time series dataset. Hence, time series analysis is necessary to help uncover more trends, seasonality, repeated lag dependence, and stationarity (36).

### 7.1 Pedestrian Count Over Time (24-Hour Rolling Mean)

**Justification**

The previous plot with the pedestrian count over time was noisy, so smoothing over 24 hours may help reveal more patterns of pedestrian activity without the hour-to-hour volatility (40).

**Output Explanation**

From the plot, it became much more obvious that the highest peaks were during New Year's, where pedestrian visit the Melbourne CBD to see the fireworks, and confirms what has been previously discussed. There does seem to be a slightly increasing trend, which suggests that the influence of COVID-19 is still recovering.

### 7.2 Seasonal Pattern (24-Hour Cycle)

**Justification**

Seasonal decomposition is necessary because hourly pedestrian demand is expected to have a strong daily cycle, so checking the seasonal component can help check how the typical hour of the day affects pedestrian activity. Since people usually move through the city at different levels during the morning, workday, evening, and late night, this step helps show whether the hour of the day is a factor in pedestrian demand (41).

**Output Explanation**

From the plot, it is clear that the seasonal effect is strongly negative at night and becomes strongly positive during the workday, especially the peak at 5 PM, when most people finish work and are looking to go home. The lowest tends to be around 3 AM, which is expected, since people tend to party late until 12 PM before heading home, and the other reason is that the city's pedestrian activities are very limited at that time. This plot further confirms that hour-of-day is a major driver of pedestrian demand, and there may be other variables influencing pedestrian demand.

### 7.3 Autocorrelation Of Pedestrian Count

**Justification**

Checking for autocorrelation is important because it shows whether the current pedestrian demand depends on previous hours, and if strong dependence exists, then lag features will be very useful for forecasting (42).

**Output Explanation**

From the plot, there appears to be a very strong repeating pattern at a regular interval, especially around daily cycles. This repeated structure applies not only on the daily level, but also appears to be on the weekly level as well, meaning daily and weekly lag features will be useful for predictions.

### 7.4 Augmented Dickey-Fuller Test

**Justification**

The Augmented Dickey-Fuller Test checks whether the time series is statistically stationary in the unit-root sense, and while not necessary for deep learning modelling, it does provide some more understanding of the patterns underlying the merged dataset (43).

**Output Explanation**

The result was that the p-value is extremely small and the test statistic is far below the critical values, hence, the null hypothesis of a unit root is rejected. This means that the time series dataset is statistically stationary enough to exhibit a learnable structure rather than behaving like a random walk, which basically means that it's not necessarily a random coin toss in layman's terms.

## 8. Feature Engineering

The feature engineering section is necessary because the current variables in the merged datasets may not necessarily be enough for a strong forecasting model, so doing feature engineering will create more useful variables that help capture other aspects of the datasets, like cyclical structures, recent history, and short-term trends (44, 45).

### 8.1 Creating Time Features

**Justification**

Creating more calendar-based features is necessary because pedestrian activity depends on when that observation happened, as shown in previous plots. Hence, hours, day of week, month, and weekend status are all useful predictors. This step ensures the datetime_hour column is converted into its individual components (45).

**Output Explanation**

The output shows that new time-based columns were added to the dataset. These include hour, day_of_week, month, and is_weekend.

### 8.2 Creating Cyclical Time Features

**Justification**

Cyclical encoding is necessary since time variables are cyclical and not linear, like a clock. If we're talking just normal values like 0 and 23, these two values are quite far apart, but it's not, since it's time and there's only 1 hour difference. Using sine and cosine preserves that circular structure. This step ensures that the time variables are cyclical and prevents the models from learning misleading distances between values at the edge of a cycle (46).

**Output Explanation**

The output shows that new cyclical time features were created, including hour_sin, hour_cos, dow_sin, dow_cos, month_sin, and month_cos.

### 8.3 Creating Cyclical Wind Direction Features

**Justification**

Wind direction is also circular, being 360 degrees, meaning 0 and 359 degrees are almost the same direction. So wind direction must also be converted to cyclical for a continuous form, and ensure consistency with treatments of cyclical variables like time (46).

**Output Explanation**

The output shows that two new features were created, wind_dir_sin and wind_dir_cos.

### 8.4 Creating Lag Features

**Justification**

Lag features are important, as indicated by the autocorrelation plot, since pedestrian demand is highly dependent on recent history based on the autocorrelation plot (47, 48).

**Output Explanation**

The features lag_1, lag_24, and lag_168 capture the previous hour, previous day, and previous week at the same hour. This does create missing values since there wasn't recent information to populate the lag columns for specific rows, which will be dealt with. Like the very first row having a missing value for lag_1, because there wasn't a previous row to populate that value.

### 8.5 Creating Rolling Features

**Justification**

Rolling features summarise recent pedestrian counts rather than relying on a single observation for a datapoint, allowing the model to capture the short-term trend and help smooth the short-term noise (47).

**Output Explanation**

Some new predictors were added, like rolling_mean_24, which summarises the previous 24 hours, while rolling_mean_168 summarises the previous week. And as expected, similar to lag features but not the same, the first rows are missing values since the rolling window cannot be calculated until enough history exists. 

### 8.6 Removing NA Rows

**Justification**

This step removes the rows with missing values created from the lag and rolling features at the beginning of the ordered merged dataset, since they cannot be used as they are incomplete predictor information. By removing these rows, the dataset lost a part of the early period as a trade-off in the pipeline, which is reasonable considering the dataset is being updated in real-time, so there will be more data points to use in the future, hence, negligible (48).

**Output Explanation**

The output shows that all rows with missing values were removed, and the dataset was reset into a clean index.

### 8.7 Removing Unnecessary Features

**Justification**

Since the cyclical variables were created, the original raw cyclical variables have become redundant. So removing these variables helps reduce feature duplication and makes the modelling easier and cleaner (23, 24).

**Output Explanation**

The averagewinddirection, hour, day_of_week, and month columns are dropped. Resulting in 24 columns, including the target pedestrian count, selected climate variables, is_weekend, cyclical encodings, lag features, and rolling means. The datetime_hour is temporarily kept, but will be removed later on as well.

## 9. Preparing Train/Val/Test Splits

Preparing the splits is necessary for training a forecasting model, so that the future periods are evaluated, and not on randomly selected time periods. Hence, splitting the dataset based on time is important so that the model being trained on the past can predict the future, like in real-world scenarios (49).

### 9.1 Splitting The Data By Time

**Justification**

Chronological splitting in time-series forecasting is to prevent leaking future information into the training process. For example is if random splitting were to be used, then a model can be trained on data points in 2026, but is tested on a time period in 2025. Hence, the validation and test set must be later, and the training set must be the past, as shown here (49).

The split for training, validation and testing is 80:10:10 to ensure that there are enough data points used for training, with enough data points to perform validation and testing. Deep learning tends to require a large number of data points, so using this split ratio prevents underfitting (50).

**Output Explanation**

The output confirms that the data was split chronologically, with the training data coming first, followed by validation and testing data.

### 9.2 Separating Features And Target

**Justification**

This step is necessary to separate the predictor inputs X and the output target y. This is necessary because the model needs to know which columns are used as inputs and which column it needs to predict (51).

**Output Explanation**

The target is set to pedestriancount, which is the variable the model is trying to predict. The datetime column is excluded from the features since the chronological splitting is completed, so this leaves 22 predictor columns for each split, as shown in the shapes output.

### 9.3 Scaling The Features

**Justification**

Scaling the features is necessary because the predictor variables are measured on different scales. For example, temperature, humidity, air pressure, wind speed, pollution values, and lagged pedestrian counts all have different ranges. Hence, needs to be standardised so that the variables are unitless, allowing the variables to be able to directly compared (52).

Standardisation basically sets the predictors to have a mean of 0 and a standard deviation of 1. This allows faster convergence when all the input features are on the same scale, preventing feature dominance due to differences in magnitudes, and is generally more stable (52).

**Output Explanation**

The output shows that the training, validation, and test feature sets were successfully scaled using StandardScaler.

### 9.4 Creating LSTM Sequences

**Justification**

Creating LSTM sequences is necessary because an LSTM model needs data in a sequential format, rather than as separate individual rows. Since this project uses hourly time-series data, the model should look at a window of past observations before predicting the current pedestrian count. A sequence length of 24 hours is used so the model can learn from the previous full day of pedestrian, climate, and time-based patterns (53).

**Output Explanation**

The output shows that the scaled training, validation, and test data were converted into 3D LSTM input sequences.

## 10. Baseline LSTM Model

A baseline LSTM model will be built with randomly specified settings to set the baseline model performance to compare with later models using hyperparameter tuning. Basically helps to decide whether later optimisation improves the model's performance (54, 55).

### 10.1 Model Building

**Justification**

The baseline model uses the Adam optimiser because it adjusts the learning rate during training, which makes it efficient and stable for deep learning tasks (56). The MSE loss function is used because this is a regression problem where the model is predicting a numeric pedestrian count (57). Early stopping is also included to prevent unnecessary training once the validation loss stops improving, since performance may worsen once it plateaus due to overfitting, resulting in worse performance on validation sets (58).

**Output Explanation**

The training loss and validation loss both decrease over time, which means that the model is learning patterns from the training data (59). It does reach a plateau at a certain point, and the validation loss starts going up a bit. Hence, the early stopping happened, putting a stop to the training process and restoring the lowest validation loss model weights (58, 59).

### 10.2 Baseline Loss Plot

**Justification**

Plotting the baseline LSTM model is necessary to see how the model was performing throughout the training process. The training loss curve shows how well it's learning from the training data, and the validation loss curve shows how the model trained on the training data performs on the validation data that it hasn't seen before (59).

**Output Explanation**

The output shows the training loss and validation loss across each epoch for the baseline LSTM model, with the training loss still going down, but early stopping happened since the validation loss was starting to go back up, meaning overfitting was starting to happen (58, 59).

## 11. Hyperparameter Tuning LSTM Model

Hyperparameter tuning is necessary because the baseline LSTM model may not use the best settings for this dataset. Different choices, such as the number of LSTM units, dropout rate, learning rate, and batch size, can affect how well the model learns from the time-series data (60, 61, 62, 63). This will ensure the evaluation metrics for the model's performance will be higher, avoiding overfitting/underfitting, and allowing faster convergence (60, 61, 62).

### 11.1 Model Building With Hyperparameter Comparisons

**Justification**

A small search space was used to compare multiple combinations, just to see what works best without using up too much computational power and time (60). The tuning grid includes different numbers of LSTM units, dropout rates, learning rates, and batch sizes. These hyperparameters were selected because more LSTM units can help the model learn more complex time-series patterns, dropout can reduce overfitting by randomly ignoring some neurons during training, the learning rate controls how quickly the model updates during training, and batch size affects training speed and model stability (60, 61, 62, 63).

**Output Explanation**

Multiple different combinations of hyperparameters were used to compare validation RMSE performance, which was stored, then sorted to determine the best performing settings with the lowest RMSE. Overall, the tuning result shows that adding dropout and increasing the LSTM units improved performance compared to the baseline settings. This tuned model is then saved for final comparison against the baseline model.

## 12. Evaluating Model Performance

Evaluating a model's performance is important to see how well the model might work in the real-world. Hence, having some evaluation metrics ensures that this process is objective and comparable (65). Metrics such as MAE, RMSE, and R-squared help show how close the predictions are to the actual pedestrian counts and how well the model explains the overall variation in the target variable (64, 65, 66, 67).

### 12.1 Baseline Vs Best Hyperparameter Tuning Model

**Justification**

The evaluation uses MAE, RMSE, and R-squared because they are suitable metrics for a regression forecasting task (65). MAE measures the average absolute prediction error, so it shows how far the predicted pedestrian counts are from the actual values on average, so a lower MAE is better (66). RMSE also measures prediction error, but it gives larger errors more weight. This is useful because big mistakes in pedestrian demand forecasting are more important to detect, so a lower RMSE is better (64). R-squared measures how much of the variation in pedestrian count is explained by the model, this helps show the overall fit of the model, so a higher R-squared is better (67).

**Output Explanation**

The tuned model has a lower MAE than the baseline model, has a lower RMSE, and a higher R-squared. This shows that the tuned LSTM performs better than the baseline LSTM on the test dataset.

### 12.2 Actual Vs Predicted Values Plot Comparisons

**Justification**

To provide a visual representation of how the predicted values of the baseline and tuned LTSM models match up against the actual test values, and to see how well they align with the actual test values, which might have resulted in the difference in the evaluation metrics score (68, 69).

**Output Explanation**

From the plot, both models follow the curves of actual values pretty well, meaning they are able to learn the main pedestrian demand pattern. However, the tuned LSTM appears to follow the actual values just a tad more closely than the baseline model.

## 13. Using The Best Model

Selecting the best model is necessary because the final prediction should use the model that performed the best during testing. Testing with the best model is an enactment of how the model might perform in the real world, and seeing firsthand how similar the results would be.

### 13.1 Predicting With The Best Model

**Justification**

The best model was the fine-tuned model, so the final prediction will be made with it for the final forecasting step by using the latest available data and predicting the pedestrian count for that time.

**Output Explanation**

The output shows that the Tuned LSTM was selected as the best model, as mentioned earlier.

### 13.2 Preparing The Latest Input Window

**Justification**

Preparing the latest input window is necessary because the LSTM model needs a sequence of recent observations before it can make a prediction. Since the model was trained using 24-hour sequences, the latest 24 rows of feature data need to be selected and formatted in the same way. The latest feature window is also scaled using the same scaler from training, which keeps the input format consistent with what the model has already learned.

**Output Explanation**

The output shows that the latest input window was reshaped into 1 sequence, 24 time steps, and 22 features. Ready to be used for prediction.

### 13.3 Making The Prediction

**Justification**

The model predicts the target in scaled form, so the prediction needs to be converted back to the original pedestrian count scale. This is because the LSTM model was trained using the scaled target variable instead of the original pedestrian count values. The result is also rounded because the pedestrian count should be presented as a whole number, since you can't exactly have half or parts of a person.

**Output Explanation**

The output shows the latest timestamp, the actual pedestrian count, and the predicted pedestrian count. The predicted value is close to the actual value, which suggests that the tuned LSTM model is able to make a reasonable estimate using the most recent 24-hour input sequence.

### 13.4 Bar Chart Of Actual Vs Predicted

**Justification**

A bar chart was included just to have a visual comparison of the tuned LTSM model's prediction against the actual value.

**Output Explanation**

From the chart, the predicted value is slightly lower than the actual value, meaning the model underestimated the pedestrian count for the latest timestamp. However, the two bars are still relatively close, which suggests that the model produced a reasonable prediction.

Though the difference may possibly be due to predictors that were unaccounted for, as pedestrian activity depends on many other factors, like public holidays, WFH policies, and maybe even geopolitical events like the Iran War happening as of 2026.

## 14. References

1. https://pandas.pydata.org/
2. https://pandas.pydata.org/docs/user_guide/timeseries.html
3. https://matplotlib.org/
4. https://scikit-learn.org/stable/modules/preprocessing.html
5. https://www.tensorflow.org/guide/keras
6. https://peps.python.org/pep-0008/
7. https://www.tensorflow.org/api_docs/python/tf/keras/utils/set_random_seed
8. https://help.opendatasoft.com/apis/ods-explore-v2/
9. https://data.melbourne.vic.gov.au/explore/dataset/pedestrian-counting-system-sensor-locations/
10. https://data.melbourne.vic.gov.au/explore/dataset/pedestrian-counting-system-monthly-counts-per-hour/
11. https://data.melbourne.vic.gov.au/explore/dataset/microclimate-sensors-data/
12. https://www.ibm.com/think/topics/data-profiling
13. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shape.html
14. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.columns.html
15. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dtypes.html
16. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html
17. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html
18. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html
19. https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
20. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nunique.html
21. https://www.ibm.com/think/topics/data-cleaning
22. https://www.tableau.com/learn/articles/what-is-data-cleaning
23. https://aml4td.org/chapters/feature-selection.html
24. https://www.ibm.com/think/topics/feature-selection
25. https://arxiv.org/html/2404.04905v1
26. https://cribl.io/glossary/data-normalization/
27. https://www.epirhandbook.com/en/new_pages/dates.html
28. https://skforecast.org/0.12.1/faq/time-series-aggregation
29. https://support.microsoft.com/en-us/office/join-tables-and-queries-3f5838bd-24a0-4832-9bc1-07061a1478f6
30. https://www.ibm.com/think/topics/data-integration
31. https://www.rudderstack.com/learn/data-collection/validation-of-data-collection/
32. https://cloud.google.com/learn/what-is-time-series
33. https://www.mdpi.com/2624-6511/8/3/82
34. https://www.itl.nist.gov/div898/handbook/eda/section1/eda11.htm
35. https://otexts.com/fpp2/tspatterns.html
36. https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm
37. https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm
38. https://www.itl.nist.gov/div898/handbook/eda/section3/histogra.htm
39. https://online.stat.psu.edu/stat200/lesson/3/3.4/3.4.2
40. https://statisticsbyjim.com/time-series/moving-averages-smoothing/
41. https://www.datacamp.com/tutorial/time-series-decomposition
42. https://online.stat.psu.edu/stat462/node/188/
43. https://support.minitab.com/en-us/minitab/help-and-how-to/statistical-modeling/time-series/how-to/augmented-dickey-fuller-test/interpret-the-results/all-statistics-and-graphs/
44. https://www.ibm.com/think/topics/feature-engineering
45. https://dotdata.com/blog/practical-guide-for-feature-engineering-of-time-series-data/
46. https://skforecast.org/latest/faq/cyclical-features-time-series.html
47. https://feature-engine.trainindata.com/en/1.8.x/user_guide/timeseries/forecasting/index.html
48. https://feature-engine.trainindata.com/en/1.8.x/user_guide/timeseries/forecasting/LagFeatures.html
49. https://apxml.com/courses/time-series-analysis-forecasting/chapter-6-model-evaluation-selection/train-test-split-time-series
50. https://www.v7labs.com/blog/train-validation-test-set
51. https://www.ibm.com/think/topics/supervised-learning
52. https://apxml.com/courses/introduction-to-neural-networks/chapter-2-data-preparation-neural-networks/feature-scaling
53. https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
54. https://blog.ml.cmu.edu/2020/08/31/3-baselines/
55. https://www.tensorflow.org/tutorials/structured_data/time_series
56. https://keras.io/api/optimizers/adam/
57. https://apxml.com/courses/introduction-to-deep-learning/chapter-3-training-loss-optimization/regression-loss-functions
58. https://srdas.github.io/DLBook/ImprovingModelGeneralization.html
59. https://developers.google.com/machine-learning/crash-course/overfitting/interpreting-loss-curves
60. https://aws.amazon.com/what-is/hyperparameter-tuning/
61. https://jmlr.org/papers/v15/srivastava14a.html
62. https://www.ibm.com/think/topics/learning-rate
63. https://machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-neural-networks-with-gradient-descent-batch-size/
64. https://www.datacamp.com/tutorial/rmse
65. https://apxml.com/courses/getting-started-with-scikit-learn/chapter-2-supervised-learning-regression/regression-evaluation-metrics
66. https://www.datacamp.com/tutorial/mean-absolute-error
67. https://www.datacamp.com/tutorial/r-squared
68. https://www.mathworks.com/help/stats/assess-model-performance-in-regression-learner.html
69. https://www.sciencedirect.com/science/article/abs/pii/S0304380008002305
