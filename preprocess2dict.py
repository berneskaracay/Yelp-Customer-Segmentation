# +

from __future__ import print_function, division
from builtins import range, input

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# +
# load in the data

df = pd.read_csv('data//small_rating.csv')
# -

N = df.userId.max() + 1 # number of users
M = df.restaurant_idx.max() + 1 # number of restaurants

# split into train and test
df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

# a dictionary to tell us which users have rated which restaurants
user2restaurant = {}
# a dicationary to tell us which restaurants have been rated by which users
restaurant2user = {}
# a dictionary to look up ratings
userrestaurant2rating = {}

print("Calling: update_user2restaurant_and_restaurant2user")
count = 0
def update_user2restaurant_and_restaurant2user(row):
  global count
  count += 1
  if count % 100000 == 0:
    print("processed: %.3f" % (float(count)/cutoff))

  i = int(row.userId)
  j = int(row.restaurant_idx)
  if i not in user2restaurant:
    user2restaurant[i] = [j]
  else:
    user2restaurant[i].append(j)

  if j not in restaurant2user:
    restaurant2user[j] = [i]
  else:
    restaurant2user[j].append(i)

  userrestaurant2rating[(i,j)] = row.rating


df_train.apply(update_user2restaurant_and_restaurant2user, axis=1)

# +
# test ratings dictionary

userrestaurant2rating_test = {}
# -

print("Calling: update_userrestaurant2rating_test")
count = 0
def update_userrestaurant2rating_test(row):
  global count
  count += 1
  if count % 100000 == 0:
    print("processed: %.3f" % (float(count)/len(df_test)))

  i = int(row.userId)
  j = int(row.restaurant_idx)
  userrestaurant2rating_test[(i,j)] = row.rating
df_test.apply(update_userrestaurant2rating_test, axis=1)

userrestaurant2rating_test

# +
# note: these are not really JSONs
with open('user2restaurant.json', 'wb') as f:
  pickle.dump(user2restaurant, f)

with open('restaurant2user.json', 'wb') as f:
  pickle.dump(restaurant2user, f)

with open('userrestaurant2rating.json', 'wb') as f:
  pickle.dump(userrestaurant2rating, f)

with open('userrestaurant2rating_test.json', 'wb') as f:
  pickle.dump(userrestaurant2rating_test, f)
