# Allegheny County Housing Market Predictions [![Heroku](https://heroku-badge.herokuapp.com/?app=heroku-badge&style=flat)](https://allegheny-county-housing.herokuapp.com/)



![Pittsburgh_skyline_panorama_at_night](https://github.com/Datuashvili/Allegheny-County-Housing-Market/blob/main/Pittsburgh_skyline_panorama_at_night.jpg)



### Introduction



This project is intended to help future and present Allegheny County residents find a reasonable price for a house in Allegheny county, PA.



Whether you are buying or selling a house, App offers estimates and forecasts to help you make an informed decision.



If you are selling, buying, refinancing, or even remodeling a house in real-time, the App allows finding out what is the present value of the house based on location and other features:



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



In the second section of this App, you look up forecasted prices (for the next two years) and price trends based on zip codes. This feature could help people in family or retirement planning.  



### Gather and analyze data



I collected data from [The Western Pennsylvania Regional Data Center](https://data.wprdc.org/dataset/property-assessments).

Data went through cleaning, outlier removal, and feature engineering process.



**Heatmap:** I will use the heatmap to display the correlation between our dependant variable **'SALEPRICE'** against the main features, as well as the correlation between different independent variables:

![heatmap](https://github.com/Datuashvili/Allegheny-County-Housing-Market/blob/main/heatmap.png)





### Build Models

For House price estimation, I trained Ridge Regression Model:

* Total observations 27,398 for recently sold houses(the last two years).

* Numerical Variables: Total square footage of land, Living Space - Sqft, Stories, Total rooms, Total bedrooms, Full bathroom, Half bathroom, Fireplace.

* Categorical Variables: The exterior wall type, The roofing material type, Description for building style, The original date of construction, Location.

* Included sale date as a dependent variable.

* Test set R^2: 0.8314699491709845, Training set mean absolute error: 49688.01850678622, cross-validations score 5 folds for model: 0.83201578 0.81327462 0.81997816 0.81525159 0.82507495



For House Price Forecast and Trend:

* I selected only the last 11 years to avoid market shocks caused by the 2008 housing market bubble to have more accurate predictions.

* Total observations 131076 observations.

* Checked stationarity.

* Aggregated data by zip code. total 72 zip codes.

* Build individual SARIMAX models for each zip code.

* App allows you to compare historical prices to predicted prices.

* Finally calculate return on investment for each zip code forecast and identify zip codes that show an upward trend





Let's see how real estate prices have changed over time and will change based on predictions

<img src="https://github.com/Datuashvili/Allegheny-County-Housing-Market/blob/main/diff_time_forecast.png" width="100000" >  
