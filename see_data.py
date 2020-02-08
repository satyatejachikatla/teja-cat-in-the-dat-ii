import pandas as pd
import math

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

print('All Cols :',[col for col in train_data])
print()

## All Bins ##
print('Unique bin_0 :',set(train_data['bin_0'].unique())|set(test_data['bin_0'].unique()))
print('Unique bin_1 :',set(train_data['bin_1'].unique())|set(test_data['bin_1'].unique()))
print('Unique bin_2 :',set(train_data['bin_2'].unique())|set(test_data['bin_2'].unique()))
print('Unique bin_3 :',set(train_data['bin_3'].unique())|set(test_data['bin_3'].unique()))
print('Unique bin_4 :',set(train_data['bin_4'].unique())|set(test_data['bin_4'].unique()))

###############
print()
print()
###############
## All Norms with bins##
print('Unique nom_0 :',set(train_data['nom_0'].unique())|set(test_data['nom_0'].unique()))
print('Unique nom_1 :',set(train_data['nom_1'].unique())|set(test_data['nom_1'].unique()))
print('Unique nom_2 :',set(train_data['nom_2'].unique())|set(test_data['nom_2'].unique()))
print('Unique nom_3 :',set(train_data['nom_3'].unique())|set(test_data['nom_3'].unique()))
print('Unique nom_4 :',set(train_data['nom_4'].unique())|set(test_data['nom_4'].unique()))


#### All ranges ####
#print('Unique nom_5 :',set(train_data['nom_5'].unique())|set(test_data['nom_5'].unique()))
#print('Unique nom_6 :',set(train_data['nom_6'].unique())|set(test_data['nom_6'].unique()))
#print('Unique nom_7 :',set(train_data['nom_7'].unique())|set(test_data['nom_7'].unique()))
#print('Unique nom_8 :',set(train_data['nom_8'].unique())|set(test_data['nom_8'].unique()))
#print('Unique nom_9 :',set(train_data['nom_9'].unique())|set(test_data['nom_9'].unique()))

###############
print()
print()
###############

### Ords ####
print('Unique ord_0 :',set(train_data['ord_0'].unique())|set(test_data['ord_0'].unique()))
print('Unique ord_1 :',set(train_data['ord_1'].unique())|set(test_data['ord_1'].unique()))
print('Unique ord_2 :',set(train_data['ord_2'].unique())|set(test_data['ord_2'].unique()))
print('Unique ord_3 :',set(train_data['ord_3'].unique())|set(test_data['ord_3'].unique()))
print('Unique ord_4 :',set(train_data['ord_4'].unique())|set(test_data['ord_4'].unique()))
print('Unique ord_5 :',set(train_data['ord_5'].unique())|set(test_data['ord_5'].unique()))

#### Days Month ####
print('Unique day :',set(train_data['day'].unique())|set(test_data['day'].unique()))
print('Unique month :',set(train_data['month'].unique())|set(test_data['month'].unique()))

#### Target #######
print('Percent of targets for 1s')
print(train_data.target.sum()/len(train_data.target))

'''
#View Unique per col
for col in train_data:
	print('-------------------------------')
	print('Col:',col)
	print(train_data[col].unique())
	print('-------------------------------')
'''
'''
#order_5 is unique pairing
train_f = set()
train_s = set()
for i in list(train_data['ord_5'].unique()):
	if type(i) == type(0.0):
		train_f |= {''}
		train_s |= {''}
		continue
	train_f |= {i[0]}
	train_s |= {i[1]}

test_f = set()
test_s = set()
for i in list(test_data['ord_5'].unique()):
	if type(i) == type(0.0):
		test_f |= {''}
		test_s |= {''}
		continue
	test_f |= {i[0]}
	test_s |= {i[1]}

print(train_f^test_f)
print(train_s^test_s)
'''

