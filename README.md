# Deep learning on yelp dataset

## Executive Summary
This project aims to develop an app to help restaurants to find customer who might interest
their food. Hence, we help to decrease restaurant closing rate which is around 80 percent. For
this purpose, I will do customer segmentation by using variables like which restaurants the
customers ate before, the stars they gave the restaurants and the comments they made
before. I get the data from Yelp Open Dataset. Please use the following link to get data
https://www.yelp.com/dataset.
## Motivation
I was very surprised after I heard my favorite restaurant is going to change hands because
they just opened it one year ago. It is very bad news for a person like me who has very strong
food preferences and can only eat Turkish food. I asked them the reason why they decided to
close it. They told me that Although they spend a lot of money on advertisement, they haven&#39;t
received enough customer to run their restaurant. I made quick research. According to an Ohio
State University study published in 2005 about failed restaurants, 60 percent close or change
ownership in the first year of business, with 80 percent closing within the first five years. Those
numbers shocked me. There would be a lot of reasons for a restaurant to fail like poor
customer service, bad foods or location. Neither of those is true for the restaurant I like most,
the only reason is the inefficiency in their advertisement policy.
Hence, I decided to make an R Shiny App (or Python Bokeh) where restaurants interactively
use the app and get a set of customers who might interest the type of food they are making.
Hence, they can email or mail the advertisements to that yelp users. Moreover, they can
receive a list of Yelp reviewers who are making lots of reviews and usually giving higher points
and likes the type of food you are making. Hence, they can send those people free coupons to
try their food. I believe this kind of a Yelp App will help a lot of restaurants whose only problem
is an inefficient advertisement policy to stay in business.
## Data, Method and Strategy Solve The Problem
The Yelp dataset is a subset of our businesses, reviews, and user data for use in personal,
educational, and academic purposes. Yelp made available as JSON files, use it to teach
students about databases, to learn NLP, or for sample production data while students learn
how to make mobile apps. Data contains following information: 6,685,900 reviews, 192,609
businesses, 200,000 pictures about 10 metropolitan areas and 1,223,094 tips by 1,637,138
users. Over 1.2 million business attributes like hours, parking, availability, and ambience
Aggregated check-ins over time for each of the 192,609 businesses.

I will begin exploring the data through visualizations and code to understand how each
restaurant type is related to the others. I will observe a statistical description of the dataset,
consider the relevance of each of these features. I will check if it is possible to determine
whether customers giving a good star for one type of restaurant will necessarily imply he will
give a good star for another kind of restaurant.
I will use un-supervised machine learning to analyze yelp dataset containing a lot of data on
various yelp reviewers&#39; review count, type of restaurants they eat and reviews they give to
restaurants. One of my aim for this project is to analyze the variation in the different types of
yelp users that the restaurant owner interacts with. By doing this I would provide restaurant
owners&#39; insights into how to best structure their advertisement and coupon policy to increase
their probability stay in the competitive restaurant industry.
I also plan to do NLP on reviews column to further categorize the customers.
Finally, after I categorize the customer, I would like to use a R Shinny App (or Python Bokeh)
for customer recommendation.
