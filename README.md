# Allegheny County Housing Market Predictions [![Heroku](https://heroku-badge.herokuapp.com/?app=heroku-badge&style=flat)](https://allegheny-county-housing.herokuapp.com/)

![Pittsburgh_skyline_panorama_at_night](https://github.com/Datuashvili/Allegheny-County-Housing-Market/blob/main/Pittsburgh_skyline_panorama_at_night.jpg)

### Introduction

This project is intended to help future and present Allegheny County residents to find out what the reasonable price is for a house in Allegheny county, PA. 

Whether you are buing or selling a house App offers estimates and forecasts to help you make informed decision.

If you are selling, buying, refinancing, or even remodeling a home house in real time the App gives opportunity to find out what is the present value of the house based on location and other features:

* Total square footage of land
* Living Space - Sqft
* Stories
* Total rooms
* Total bedrooms
* Full bathroom
* Half bathroom
* Fireplace 
* Condition
* The exterior wall type
* The roofing material type
* Description for building style
* The original date of construction

If are planning future family, retirement in the second section you will be able to observ forecasted prices(for next two years) and price trend per zip-code.

### Gather and analyze data 

For project I collected data from [The Western Pennsylvania Regional Data Center](https://data.wprdc.org/dataset/property-assessments). 
Data went through cleaning, outlier removal and feature engineering process. 

**Heatmap:** let's use heatmap to Check the correlation between our dependant variable **'SALEPRICE'** against the main features:
![heatmap](https://github.com/Datuashvili/Allegheny-County-Housing-Market/blob/main/heatmap.png)


### Build Models
For House price estimation I trained Ridge Regression Model:
* Total observations 27,398 for resently old houses(last two year).
* Numerical Variables: Total square footage of land, Living Space - Sqft, Stories, Total rooms, Total bedrooms, Full bathroom, Half bathroom, Fireplace.
* Categorical Variables: The exterior wall type, The roofing material type, Description for building style, The original date of construction, Location.
* Included sale date as a dependant variable. 
* Test set R^2: 0.8314699491709845, Training set mean absolute error: 49688.01850678622, cros validations score 5 folds for model: 0.83201578 0.81327462 0.81997816 0.81525159 0.82507495

For House Price Forecast and Trend:
* I selected only last 11 years to avoid market shoks coused by 2008 housing mrket booble to have more acurate predictions
* Total observations 131076 observations
* checked stationarity
* agregated data by zip code. total 64 zip codes 
* Build individual SARIMAX models for each zip code, let's see couple of them below. 
* App gives you ability to compare historical prices to predicted prices. 

![SARIMAX](https://github.com/Datuashvili/Allegheny-County-Housing-Market/blob/main/4.png)

Let's see how the map will change based on predictions 
<img src="github.com/Datuashvili/Allegheny-County-Housing-Market/blob/main/diff_time_forecast.png " width="100" >




