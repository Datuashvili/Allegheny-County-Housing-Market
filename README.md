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
I selected only last 11 years records to avoid market shoks coused by 2008 housing market booble and have more acurate predictions. Data went through cleaning, outlier removal and feature engineering process. 

**Heatmap:** let's use heatmap to Check the correlation between our dependant variable **'SALEPRICE'** against the main features:
![heatmap](https://github.com/Datuashvili/Allegheny-County-Housing-Market/blob/main/heatmap.png)


