import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import regularizers
import numpy as np
import time
import pandas as pd

save_folder='./normalized_data/'
print('Loading::::::')
train_X = np.array(pickle.load( open( save_folder+"train_X", "rb" ) ))
print('Loaded train_X')
train_Y = np.array(pickle.load( open( save_folder+"train_Y", "rb" ) ))
print('Loaded train_Y')

NAME = "Cats{}".format(int(time.time()))
print('train_X',train_X.shape)
print('train_Y',train_Y.shape)

sizes = [16,]
n_dense = [3]
for s in sizes:
	for d in n_dense: 	 
		model = tf.keras.models.Sequential()
		model.add(tf.keras.layers.Flatten())
		
		for i in range(d):
			model.add(tf.keras.layers.Dense(s,activation=tf.nn.relu,kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l1(0.001)))
		model.add(tf.keras.layers.Dense(1))#,activation=tf.nn.sigmoid))

		model.compile(optimizer=tf.keras.optimizers.Adam(),
					  loss='MSE',
					  metrics=['accuracy']
					  )
		tensorboard = TensorBoard(log_dir="./logs/{}-size-{}-n_dense-{}".format(NAME,s,d))
		print('Model Compiled')

		model.fit(train_X,train_Y,epochs=10,batch_size=16,
			validation_split=0.1,callbacks=[tensorboard],
			)
		print('Model fitted')

		model.save('models/abc')