# +
from __future__ import print_function, division
from builtins import range, input


import pickle
import numpy as np
import pandas as pd
from collections import Counter

# +
# load in the data

df = pd.read_csv('data/edited_rating.csv')
print("original dataframe size:", len(df))
# -

N = df.userId.max() + 1 # number of users
M = df.restaurant_idx.max() + 1 # number of restaurants

print("count users and restaurants: ", N,"and",M)

user_ids_count = Counter(df.userId)
restaurant_ids_count = Counter(df.restaurant_idx)

# +
# number of users and restaurants we would like to keep
n =10000
m = 2000

user_ids = [u for u, c in user_ids_count.most_common(n)]
restaurant_ids = [m for m, c in restaurant_ids_count.most_common(m)]
# -

df.shape

# make a copy, otherwise ids won't be overwritten
df_small = df[df.userId.isin(user_ids) & df.restaurant_idx.isin(restaurant_ids)]

df_small.shape

# +
# need to remake user ids and restaurant ids since they are no longer sequential
new_user_id_map = {}
i = 0
for old in user_ids:
  new_user_id_map[old] = i
  i += 1
print("i:", i)

new_restaurant_id_map = {}
j = 0
for old in restaurant_ids:
  new_restaurant_id_map[old] = j
  j += 1
print("j:", j)

# +
print("Setting new ids")
df_small.loc[:, 'userId'] = df_small.apply(lambda row: new_user_id_map[row.userId], axis=1)
df_small.loc[:, 'restaurant_idx'] = df_small.apply(lambda row: new_restaurant_id_map[row.restaurant_idx], axis=1)
# df_small.drop(columns=['userId', 'restaurant_idx'])
# df_small.rename(index=str, columns={'new_userId': 'userId', 'new_restaurant_idx': 'restaurant_idx'})
print("max user id:", df_small.userId.max())
print("max restaurant id:", df_small.restaurant_idx.max())

print("small dataframe size:", len(df_small))
df_small.to_csv('data/small_rating.csv', index=False)
# -




