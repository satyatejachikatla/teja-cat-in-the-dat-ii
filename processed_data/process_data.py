import pandas as pd
import math

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

print('All Cols :',[col for col in train_data])

def clean_nan(d):
	for i in range(len(d)):
		if math.isnan(d[i]):
			d[i]=0
		else:
			d[i]=int(d[i])

train_days = {'day':list(train_data['day'])}
train_months ={'month': list(train_data['month'])}

test_days = {'day':list(test_data['day'])}
test_months = {'month':list(test_data['month'])}

#Cleaning nans
clean_nan(train_days['day'])
clean_nan(train_months['month'])
clean_nan(test_days['day'])
clean_nan(test_months['month'])

train_targets = {'target':list(train_data['target'])}

## All Bins ##
bins = ('bin_0','bin_1','bin_2','bin_3','bin_4')
train_bins = { col:[] for col in bins}
test_bins  = { col:[] for col in bins}
def convert_bins_row_to_int_arr(row):
	if type(row) == type(''):
		if row == 'Y' or row == 'T':
			return [1,1]
		if row == 'N' or row == 'F':
			return [0,1]
	if type(row) == type(0.0):
		if math.isnan(row):
			return [0,0]
		else:
			return [int(row),1]

for col in bins:
	for row in train_data[col]:
			train_bins[col].append(convert_bins_row_to_int_arr(row))
	for row in test_data[col]:
			test_bins[col].append(convert_bins_row_to_int_arr(row))

## Norm 0-4 ##
noms04 = ('nom_0','nom_1','nom_2','nom_3','nom_4')
train_noms04 = { col:[] for col in noms04}
test_noms04  = { col:[] for col in noms04}
def convert_row_into_one_hot_noms04(row,label_format):
	if type(row) == type(0.0):
		row = 'nan'
	one_hot = [ 0 for _ in range(len(label_format)) ]
	one_hot[label_format.index(row)] = 1
	return one_hot
for col in noms04:
	label_format = sorted(set( 'nan' if type(i) == type(0.0) else i  for i in train_data[col].unique()))
	for row in train_data[col]:
		train_noms04[col].append(convert_row_into_one_hot_noms04(row,label_format))
	for row in test_data[col]:
		test_noms04[col].append(convert_row_into_one_hot_noms04(row,label_format))


#### Norms 5-9 ####
noms59 = ('nom_5','nom_6','nom_7','nom_8','nom_9')
train_noms59 = { col:[] for col in noms59}
test_noms59  = { col:[] for col in noms59}
def convert_row_into_one_hot_noms59(row):
	if type(row) == type(0.0):
		row = '0'
	one_hot = [0 for _ in range(36)]

	for i,c in enumerate(bin(int('0x'+row,0))[2:]):
		one_hot[i] = int(c)

	return one_hot
for col in noms59:
	for row in train_data[col]:
		train_noms59[col].append(convert_row_into_one_hot_noms59(row))
	for row in test_data[col]:
		test_noms59[col].append(convert_row_into_one_hot_noms59(row))

#### Ord 0-4 ####
ord04 = ('ord_0','ord_1','ord_2','ord_3','ord_4')
train_ord04 = { col:[] for col in ord04}
test_ord04  = { col:[] for col in ord04}
def convert_row_into_one_hot_ord04(row,label_format):
	row = str(row)
	one_hot = [ 0 for _ in range(len(label_format)) ]
	one_hot[label_format.index(row)] = 1
	return one_hot

for col in ord04:
	label_format = sorted(set( str(i) for i in train_data[col].unique()))
	for row in train_data[col]:
		train_ord04[col].append(convert_row_into_one_hot_ord04(row,label_format))
	for row in test_data[col]:
		test_ord04[col].append(convert_row_into_one_hot_ord04(row,label_format))

### Ord5 ####
ord5='ord_5'
train_ord5 = {ord5:[]}
test_ord5  = {ord5:[]}
l1 = sorted({ '' if str(i) == 'nan' else i[0] for i in train_data[ord5].unique() })
l2 = sorted({ '' if str(i) == 'nan' else i[1] for i in train_data[ord5].unique() })
def convert_row_into_one_hot_ord5(row,l1,l2):
	row = str(row)
	o_l1 = [ 0 for _ in range(len(l1)) ]
	o_l2 = [ 0 for _ in range(len(l2)) ]

	if row == 'nan':
		o_l1[l1.index('')] = 1
		o_l2[l2.index('')] = 1
	else:
		o_l1[l1.index(row[0])] = 1
		o_l2[l2.index(row[1])] = 1
	return o_l1 + o_l2
for row in train_data[ord5]:
	train_ord5[ord5].append(convert_row_into_one_hot_ord5(row,l1,l2))
for row in test_data[ord5]:
	test_ord5[ord5].append(convert_row_into_one_hot_ord5(row,l1,l2))

### Making itterable df ###
train_df = pd.DataFrame({**train_bins,**train_noms04,**train_noms59,**train_ord04,**train_ord5,**train_days,**train_months})
test_df = pd.DataFrame({**test_bins,**test_noms04,**test_noms59,**test_ord04,**test_ord5,**test_days,**test_months})
'''
train_df.to_csv('./train_temp.cvs')
test_df.to_csv('./test_temp.cvs')
'''
## Segregating X and Y ##
train_X = []
for _,row in train_df.iterrows():
	train_X.append([])
	for col in ('bin_0','bin_1','bin_2','bin_3','bin_4',
				'nom_0','nom_1','nom_2','nom_3','nom_4',
				'nom_5','nom_6','nom_7','nom_8','nom_9',
				'ord_0','ord_1','ord_2','ord_3','ord_4',
				'ord_5'):
		train_X[-1].extend(row[col])
	train_X[-1].append(row['day'])
	train_X[-1].append(row['month'])
train_Y = train_targets['target']

test_X = []
for _,row in test_df.iterrows():
	test_X.append([])
	for col in ('bin_0','bin_1','bin_2','bin_3','bin_4',
				'nom_0','nom_1','nom_2','nom_3','nom_4',
				'nom_5','nom_6','nom_7','nom_8','nom_9',
				'ord_0','ord_1','ord_2','ord_3','ord_4',
				'ord_5'):
		test_X[-1].extend(row[col])
	test_X[-1].append(row['day'])
	test_X[-1].append(row['month'])

print(test_X[1],len(test_X[1]))

# Saving Progress #
save_dir ='./processed_data/'
import os
try:
	os.mkdir(save_dir)
except FileExistsError:
	pass

import pickle
#pickle.dump(train_X, open( save_dir +"train_X", "wb" ))
#pickle.dump(train_Y, open( save_dir +"train_Y", "wb" ))
pickle.dump(test_X, open( save_dir +"test_X", "wb" ))

