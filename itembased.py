# +

from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future
# -

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList

# +
# load in the data
import os
if not os.path.exists('user2restaurant.json') or \
   not os.path.exists('restaurant2user.json') or \
   not os.path.exists('userrestaurant2rating.json') or \
   not os.path.exists('userrestaurant2rating_test.json'):
   import preprocess2dict

with open('user2restaurant.json', 'rb') as f:
  user2restaurant = pickle.load(f)

with open('restaurant2user.json', 'rb') as f:
  restaurant2user = pickle.load(f)

with open('userrestaurant2rating.json', 'rb') as f:
  userrestaurant2rating = pickle.load(f)

with open('userrestaurant2rating_test.json', 'rb') as f:
  userrestaurant2rating_test = pickle.load(f)
# -

N = np.max(list(user2restaurant.keys())) + 1
# the test set may contain restaurants the train set doesn't have data on
m1 = np.max(list(restaurant2user.keys()))
m2 = np.max([m for (u, m), r in userrestaurant2rating_test.items()])
M = max(m1, m2) + 1
print("N:", N, "M:", M)

# +
if M > 2000:
  print("N =", N, "are you sure you want to continue?")
  print("Comment out these lines if so...")
  exit()

# to find the user similarities, you have to do O(M^2 * N) calculations!
# in the "real-world" you'd want to parallelize this
# note: we really only have to do half the calculations, since w_ij is symmetric
K = 20 # number of neighbors we'd like to consider
limit = 5 # number of common restaurants users must have in common in order to consider
neighbors = [] # store neighbors in this list
averages = [] # each item's average rating for later use
deviations = [] # each item's deviation for later use
# -

for i in range(M):
  # find the K closest items to item i
  users_i = restaurant2user[i]
  users_i_set = set(users_i)

  # calculate avg and deviation
  ratings_i = { user:userrestaurant2rating[(user, i)] for user in users_i }
  avg_i = np.mean(list(ratings_i.values()))
  dev_i = { user:(rating - avg_i) for user, rating in ratings_i.items() }
  dev_i_values = np.array(list(dev_i.values()))
  sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

  # save these for later use
  averages.append(avg_i)
  deviations.append(dev_i)

  sl = SortedList()
  for j in range(M):
    # don't include yourself
    if j != i:
      users_j = restaurant2user[j]
      users_j_set = set(users_j)
      common_users = (users_i_set & users_j_set) # intersection
      if len(common_users) > limit:
        # calculate avg and deviation
        ratings_j = { user:userrestaurant2rating[(user, j)] for user in users_j }
        avg_j = np.mean(list(ratings_j.values()))
        dev_j = { user:(rating - avg_j) for user, rating in ratings_j.items() }
        dev_j_values = np.array(list(dev_j.values()))
        sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

        # calculate correlation coefficient
        numerator = sum(dev_i[m]*dev_j[m] for m in common_users)
        w_ij = numerator / (sigma_i * sigma_j)

        # insert into sorted list and truncate
        # negate weight, because list is sorted ascending
        # maximum value (1) is "closest"
        sl.add((-w_ij, j))
        if len(sl) > K:
          del sl[-1]

  # store the neighbors
  neighbors.append(sl)

  # print out useful things
  if i % 1 == 0:
    print(i)


def predict(i, u):
  # calculate the weighted sum of deviations
  numerator = 0
  denominator = 0
  for neg_w, j in neighbors[i]:
    # remember, the weight is stored as its negative
    # so the negative of the negative weight is the positive weight
    try:
      numerator += -neg_w * deviations[j][u]
      denominator += abs(neg_w)
    except KeyError:
      # neighbor may not have been rated by the same user
      # don't want to do dictionary lookup twice
      # so just throw exception
      pass

  if denominator == 0:
    prediction = averages[i]
  else:
    prediction = numerator / denominator + averages[i]
  prediction = min(5, prediction)
  prediction = max(0.5, prediction) # min rating is 0.5
  return prediction


train_predictions = []
train_targets = []
for (u, m), target in userrestaurant2rating.items():
  # calculate the prediction for this restaurant
  prediction = predict(m, u)

  # save the prediction and target
  train_predictions.append(prediction)
  train_targets.append(target)

test_predictions = []
test_targets = []
# same thing for test set
for (u, m), target in userrestaurant2rating_test.items():
  # calculate the prediction for this restaurant
  prediction = predict(m, u)

  # save the prediction and target
  test_predictions.append(prediction)
  test_targets.append(target)


# +
# calculate accuracy
def mse(p, t):
  p = np.array(p)
  t = np.array(t)
  return np.mean((p - t)**2)

print('train mse:', mse(train_predictions, train_targets))
print('test mse:', mse(test_predictions, test_targets))
