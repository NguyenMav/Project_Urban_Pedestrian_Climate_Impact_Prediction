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

**Justification**

This section is to show what libraries were used for this use case, with each imported library supporting a specific part of the pipeline. These libraries are necessary for doing data handling, time-series analysis, visualisation, feature engineering, optimisation, and deep learning. These were added at the beginning to ensure the workflow is organised. A random seed was set with a student ID to allow reproducibility of the outputs in this notebook.

## 2. Importing The Datasets

**Justification**

This section is necessary for importing multiple public datasets from the City of Melbourne, before any cleaning, merging and modelling in later stages can happen. These datasets will be accessed through the City of Melbourne Open Data API v2.1, to allow the notebook to be directly used upon download.

By using a shared BASE_URL and a dictionary of dataset identifiers, this allows for removing and adding datasets more easily. The get_csv_url() function is especially useful because it standardises the dataset access method and allows the same logic to be reused across all three datasets. The ROW_LIMIT parameter was added to allow experimentation on smaller samples before scaling to the full dataset. Lastly, the datasets are accessed through the Melbourne Open Data API v2.1, and no visible API key is exposed in the code.

### 2.1 Pedestrian Counting System - Sensor Locations

**Justification**

This dataset was included to provide contextual information about the physical pedestrian counting network, even though it wasn't used in the later steps for modelling. The sensor metadata helps explain where the mobility data originates from and what the coverage of the system looks like.

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

This is the target dataset for the use case because the final prediction task is to forecast pedestrian demand. The pedestrian dataset contains the observed mobility outcome that the model aims to learn.

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

This dataset provides the microclimate data that the use case requires to understand how climate conditions affect urban pedestrian movement, more specifically, the input features. It basically provides the explanatory environmental variables to connect weather conditions to mobility demand. 

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
