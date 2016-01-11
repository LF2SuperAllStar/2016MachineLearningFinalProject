from sklearn.ensemble import RandomForestClassifier
import csv
from multiprocessing import Pool
import numpy as np
import time
import datetime
import random
import os

dpath = '../data/'
train_file = 'train_m.csv'
test_file = 'test_m.csv'
val_size = 19287#0
tree_num  = 30000
add_drop_rate_feature = False

def draw(data,label,val_size):
	x_y = np.array([np.append(data[n],label[n][-1]) for n in range(len(data))])
	random.shuffle(x_y)

	tr_x = x_y[val_size:][:,:-1]
	tr_y = x_y[val_size:][:,-1]
	val_x = x_y[:val_size][:,:-1]
	val_y = x_y[:val_size][:,-1]
	return tr_x,tr_y,val_x,val_y

data = np.loadtxt(dpath+train_file, delimiter=',', skiprows=1)
label = np.loadtxt(dpath+'truth_train.csv', delimiter=',')
tr_x,tr_y,val_x,val_y = draw(data,label,val_size)

print 'training at ', datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

clf = RandomForestClassifier(n_estimators=tree_num,n_jobs=-1,warm_start=True,oob_score=True)
clf.fit(tr_x,tr_y)

print 'predicting at', datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
if val_size>0:
	predict_y = clf.predict(val_x)
	print 'Eval:',1.0*[predict_y[p] == val_y[p] for p in range(len(val_y))].count(False)/len(val_y)
else:
	test_x = np.loadtxt(dpath+test_file, delimiter=',', skiprows=1,usecols=range(23))

	predict_test_y = clf.predict(test_x)
	result = sorted([[predict_test_y[t],int(test_x[t][0]),int((1+np.sign(predict_test_y[t]-0.5))/2)] for t in range(len(test_x))],reverse=True)

	submission = open('Submission.csv','w')
	for r in result: submission.write('%s,%s\n' %(r[1],r[2]))