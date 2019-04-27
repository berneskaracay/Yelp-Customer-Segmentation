from __future__ import print_function, division
from builtins import range, input
import pandas as pd
import json
import requests
from bs4 import BeautifulSoup as BS
import re
import warnings
import bokeh
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import codecs
from json import JSONDecoder
from functools import partial
pd.options.display.max_rows = 9999
pd.options.display.max_columns = 9999
warnings.filterwarnings("ignore")

business = pd.read_json('data\\business.json', lines=True)

business.shape

business.tail(2)

# +
#business["categories"].value_counts()
# -


foods_lsit=["Tapas Bars","Taiwanese","Cuban","Gluten-Free","Moroccan","African","Lebanese"",Filipino","Latin American"",Portuguese","Caribbean","Turkish","Chinese","Peruvian","Thai","Canadian (Traditional)","American (Traditional)","American (New)", "Canadian (New)","Mexican","Delis","Malaysian","Hookah","Vietnamese","Mediterranean","Specialty Food","Fish", 
            "Pakistani","Chips","Latin American","Indian","Bagels","Middle Eastern","Mexican","Ice Cream","Frozen Yogurt","Coffee & Tea","Hawaiian","Lebanese",
            "Canadian","Italian","Caribbean","Greek"," Delis","Sushi Bars ","Barbeque","Middle Eastern","Hot Dogs","Fish & Chips",
"American","Sandwiches","Coffee & Tea","Vegan","Thai","American","Burgers","Japanese","Vietnamese","French","Steakhouses","Chicken Wings","Breakfast & Brunch",
"Steakhouses","Bakeries","Sushi Bars","Juice Bars", "Smoothies","Food Trucks","Hot Dogs","Chicken Wings","Donuts","Afghan","Persian/Iranian",
"Filipino","Korean","Mediterranean","Breakfast", "Brunch","Delis","Seafood","Desserts","Sports Bars","Cafes","Ethiopian",
"Cocktail Bars","Indian","Kebab","Turkish","Arabic","Burgers","Barbeque","Vegan","Hookah Bars","Cupcakes","Soup","Southern",
"Asian","Cheesesteaks","Asian Fusion","Persian","Creperies","Iranian","Greek","French","Fast Food","Bakery","German","Brazilian","Dim Sum",
            "Seafood","Salad","Kosher","Pizza","Tapas/Small Plates","Coffee","Noodles", "Tea","Ramen","Burgers","Cajun/Creole","Tacos","Gastropubs","Tex-Mex","Burgers","Cafes","Fast Food","Diners","Seafood"]

pattern = '|'.join(foods_lsit)
pattern

business["include"]=business.categories.str.contains(pattern)

business=business.loc[(business['include']==True),:]

business=business.loc[(business['city']=='Las Vegas') ,:]

business=business.loc[(business['review_count']>=10) ,:]
business_to_include=business[["business_id"]]
business_to_include.head()

# +

business.shape
# -

for index, row in business.iterrows():
    for food in foods_lsit:
        if food in row["categories"]:
            business.loc[index,'foods']=food
            break


del business["categories"]

business.rename(columns={"foods":'categories'},inplace=True)

business = business.dropna(axis=0, subset=['categories'])

business.head()

business=business[["business_id","name","categories"]]

# ## review cleaning

review=pd.read_pickle("data\\review.pkl")

review.shape

review=pd.merge(review,business_to_include,how='inner',on='business_id')

review.head()

review=review[["business_id","user_id","stars"]]

review=pd.merge(review,business,how='inner',on='business_id')

review.rename(columns={'stars':'rating'},inplace=True)

ratings=review.copy()

ratings['restaurantId'] = ratings['business_id'].rank(method='dense').astype(int)

ratings=ratings.sort_values("restaurantId")

ratings.head()

ratings['userId'] = ratings['user_id'].rank(method='dense').astype(int)

ratings=ratings.sort_values("userId")

ratings.head()

# +
#ratings=ratings.reset_index()
# -

ratings.to_csv("data\\rating.csv")

# https://www.kaggle.com/grouplens/movielens-20m-dataset
df = pd.read_csv('data/rating.csv')

df=df.iloc[:,1:]

df.head()

# note:
# user ids are ordered sequentially from 1..138493
# with no missing numbers
# movie ids are integers from 1..131262
# NOT all movie ids appear
# there are only 26744 movie ids
# write code to check it yourself!


# make the user ids go from 0...N-1
df.userId = df.userId - 1

df["restaurant_idx"] = df["restaurantId"]-1

df.to_csv('data/edited_rating.csv', index=False)

df.head()


