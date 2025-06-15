# python-project-ApartmentPriceAnalysis
Based on the data obtained from a portal with advertisements with the sale of apartments, a model was trained that gives the forecast value of the apartment with a short report on what factors influenced the result of the model.

# Usage of the model:
* starting point in determining the price of an apartment
* quick assessment of the profitability of a given apartment (comparing the estimated price to the price in the ad)
* quick assessment of what factors influence the estimated price
* starting point for comparing several apartments or for negotiating the price

# The project consists of the following stages:
1. Transformation of data for modeling - transformation, analysis of gaps and outlier observations, selection of data for modeling and addition of data extracted from the address in the advertisement
2. Model training - appropriate coding of data, model selection and model training
3. Feature importance analysis (lokal)  - calculations with the help of a SHAP library and presentation on a graph
4. Creation of a report with a summary of the modeling result - visualization of model result and features importance

# Data used in the project:
The data for the project was taken from otodom.pl, a Polish website where users post ads for apartment sales.<br>
The data is from January 2024.<br>
The following were not used:
* data from photos 
* information provided in the verbal description given by the advertiser
* data about the distance of the apartment to various amenities (bus stops, stores, schools, etc.)<br>&nbsp;
Instead, information describing the locality (population, population density, and powiat rights of city) was added. The powiat is a unit of local government and administrative division of the second degree in Poland. A city with powiat rights is the center of such a unit
