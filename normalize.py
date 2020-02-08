import numpy as np
import pandas as pd
import pickle

save_folder='./normalized_data/'
load_folder='./processed_data/'

train_X = pickle.load( open( load_folder+"train_X", "rb" ) )
print('Loaded train_X')
train_Y = pickle.load( open( load_folder+"train_Y", "rb" ) )
print('Loaded train_Y')

max_1s = sum(train_Y)

added_0s = 0
added_1s = 0

indicies = []

for i in range(len(train_Y)):
	if train_Y[i] == 0 and added_0s < max_1s:
		added_0s += 1
		indicies.append(i)
	elif train_Y[i] == 1 and added_1s < max_1s:
		added_1s += 1
		indicies.append(i)

new_train_X = []
new_train_Y = []

from random import shuffle
shuffle(indicies)

for i in indicies:
	new_train_X.append(train_X[i])
	new_train_Y.append(train_Y[i])

import os
try:
	os.mkdir(save_folder)
except FileExistsError:
	pass

import pickle
pickle.dump(new_train_X, open( save_folder +"train_X", "wb" ))
pickle.dump(new_train_Y, open( save_folder +"train_Y", "wb" ))
