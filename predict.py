import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

model = tf.keras.models.load_model("models/abc")
save_folder='./processed_data/'
test_X  = np.array(pickle.load( open( save_folder+"test_X", "rb" ) ))
print('Loaded test_X')

test_data = pd.read_csv('./data/test.csv')
ids = {'id':list(test_data['id'])}
target = {'target':list(model.predict(test_X))}
df = pd.DataFrame({**ids,**target})

f = open('submit.csv','w')
f.write('id,target\n')
for row in df.iterrows():
	f.write(str(row[1]['id'])+','+str(list(row[1]['target'])[0])+'\n')
f.close()
#
#df.to_csv('./submit.cvs')