# Taxi Analysis 


The below an analysis I did for Kaggle’s NYC Taxi Cab Fare Prediction Competition hosted by Google Cloud and Coursera. The competition claimed that a basic model based on the distance between two points would end up with an RMSE of ~$8, so that was the target to beat. 


The data files given are as follows: 


train.csv - a file 
test.csv - a file 
sample_submission.csv - a file


The dataset consists of the following features: 


pickup_datetime - timestambe value of when the ride started
pickup_longitude - float value for the longitude of where the ride started
pickup_latitude - float value for the latitude of where the ride started
dropoff_longitdue - float value for the longitude of where the ride ended
droppoff_latitude - float value for the latitude of where the ride ended 
passenger_count - int value for the number of passengers in the taxi ride 


and then the target: 
fare_amount - flat for the dollar amount of the cost of the taxi ride 



# Exploration 
Even after dropping NA’s, there is still some nonsensical data in the data frame including: 
Negative cab fares
lat/longs of 2500+ (latitude ranges from 0 to 90, longitude from 0 to 180)
Cab rides with 0 passengers
Cars with 208 passengers

For the purpose of this analysis I limited the latitudes to between -74.2 and -73.3 and the longitudes between 40.6 and 40.9 to better capture the island of Manhattan. In addition to that I dropped all rides with that did not have 1 to 6 passengers and all rides where the fare was less than $2.5 (the base minimum fare for a cab in New York). 

First let’s look at the distribution of fares and passenger counts 

PICTURE 
PICTURE 


Even after limiting the plot to show fares < $75 there is still a heavy right skew to the distribution of fairs, with the vast majority of them below $20. Single-passenger rides are by far the most common, followed by 2-passenger and then 5-passenger rides interestingly enough.



Plotting the lat longs resulted in one of my favorite graphs I’ve ever made. Using a black background and white dots gave it the look of a satellite image of the city taken at night. Unfortunately it also revealed a problem in the data as some rides end in the East and Hudson Rivers. I could get more particular with the lat/lon exclusions but for now leave these data points in the analysis. 


PICTURE 


Now that the data has been cleaned and it looks generally fine, it’s time for some feature creation. The first features I created centered around the date and time of the ride.

PICTURE 




Extracting the Year, Month, Day of the Month, and Hour of the ride from the timestamp of the ride was simple enough with the following commands 

train_df['year'] = train_df['pickup_datetime'].dt.year
train_df['month'] = train_df['pickup_datetime'].dt.month
train_df['day'] = train_df['pickup_datetime'].dt.day
train_df['hour'] = train_df[‘pickup_datetime’].dt.hour

To see if these will help add any information to the model, I plotted the average fare by month and Year across the full time series of the data. While I was expecting modest increases due to inflation, I was completely thrown off by the massive jump in average fare in September 2012. After a little digging I found that this was a system wide price hike by yellow cabs and not a problem with the data as I originally had assumed (article: https://www.nytimes.com/2012/09/04/nyregion/new-york-taxis-to-start-charging-increased-rates.html)



PICTURE 


The next plot I made was to see if the number of passengers affected the average fare. Looking at the average fare per passenger count shows that changing the number of passengers in the cab has virtually no effect on the average price of the ride.


PICTURE 


Once I had a month variable, it was quick to map those months to seasons to see if that has any effect on the price of a cab. Using the following lines I create a season variable and then plot the average ride fare per seasons to see if that matters at all. 

season_dict = {
    12:'Winter',
    1:'Winter',
    2:'Winter',
    3:'Spring',
    4:'Spring',
    5:'Spring',
    6:'Summer',
    7:'Summer',
    8:'Summer',
    9:'Fall',
    10:'Fall',
    11:'Fall',
}
train_df['season'] = train_df['month'].map(season_dict)

While you can see that jump in september 2012 again, it appears that the season the ride took place in didn’t have much of an effect on price. 


PICTURE 



 The most important feature I created, unsurprisingly, was the distance of the ride. At first I used simple euclidian and manhattan distances, but then was turned onto Haversine distance. The Haversine formula calculates the “great-circle distance” between two point on a sphere given that the points are in longitudinal and latitudinal coordinates. Since this formula actually accounted for the curvature of the Earth, I settled on this as the final distance metric that would go into my models. 

Plotting the newly-created distance of the ride vs. the fare of the ride shows some unexpected patterns. Outside of the fact that, in general, the longer the ride the more expensive it was we see very clear horizontal lines. A little research reveled these to be airport rides which have a fixed fee that is standardized across the industry. 


PICTURE 



This discovery led to the final features I made which were the distances to the major airports (JFK, Newark, and Laguardia) which I define as the smaller of the distance between the airports and the pickup or drop-off locations of the ride. To round things out I added a similar feature to these, but this time it was the distance to the center of Manhattan which I believe would be more congested and result in longer times in the taxi and higher fares.


# Feature Selection

There is a great article for feature selection that lists 3 main waits to select features for your model: 1. Correlation plots 2. Univaraite Statistics 3. Feature Importance

#### Correlation Plot 
Below is the correlation plot of the base features along with the features I created. Not suprisingly, the distance metrics are the most correlated with the fare amount. These distances are also highly correlated with each other, which is a problem for linear regression models that can’t have high amounts of multicollinearity. 

PICTURE 


#### SelectKbest

Sklearn’s SelectKBest module will select the top features for the model, but only looks at them in isolation. It uses _______ as a metric to define feature importance. 

PICTURE 



#### Feature Importance 
A great component of decision trees and random forests is that they calculate the importance of the features given to them. After fitting Sklearn’s ExtraTreeClassifier on the data you can then extra the importance of each feature as determined by the model. 


PICTURE 


# Prediction

The two models I used in this analysis were a basic linear regression and a random forest. 

## Regression 
Since the correlation plot reveled massive correlations between the distance metrics, I’m going to only use Haversine distance for the linear regression. However, regression also assumes that variables are normally distributed. Plotting the distribution fo a few reveals that not to be the case. In these situations, it’s best to transform the variables either with log or square root transformations to coerce them to be normally distributed. Below are the same variables after I apply various transformations

PICTURE 
PICTURE 


With the variables normally distributed and not correlated with each other, I fit the linear regression model and ended up with an RMSE of $4.64 for the prediction of the validation data, a large improvement over the ~$8 in the competition’s baseline model. 

## Random Forest 

Thankfully, a random forest can handle non-normally distributed variables. I again dropped euclidian and manhattan distance from the training data as these don’t add much information past Haversine distance and then fit a random forest on the data. After some quick playing around with the tuning parameters I settled on 100 estimators and a max depth of 30. 

The result was an RMSE of $3.63 for the prediction of the validation data. This significantly outperformed the linear regression model and is far below the competition’s baseline model. 

Overall I thoroughly enjoyed working with this data and even learned some things about my city in the process. I’m extremely happy with the results I obtained, which validate all of the time spent cleaning/checking the data and creating features.


# Future Directions 
remove water points 
train on more data 
other models like xgboost 









