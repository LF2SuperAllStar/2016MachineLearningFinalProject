from sklearn.ensemble import RandomForestClassifier
import csv
from multiprocessing import Pool
import numpy as np
import time
import datetime
import random

dpath = '../data/'
val_size = 19287
tree_num  = 1000

def draw(data,label,val_size):
	x_y = np.array([np.append(data[n],label[n][-1]) for n in range(len(data))])
	random.shuffle(x_y)

	tr_x = x_y[val_size:][:,:-1]
	tr_y = x_y[val_size:][:,-1]
	val_x = x_y[:val_size][:,:-1]
	val_y = x_y[:val_size][:,-1]
	return tr_x,tr_y,val_x,val_y

print 'start at ', datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

data = np.loadtxt(dpath+'train_m.csv', delimiter=',', skiprows=1)
label = np.loadtxt(dpath+'truth_train.csv', delimiter=',')
tr_x,tr_y,val_x,val_y = draw(data,label,val_size)

print 'training at ', datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

clf = RandomForestClassifier(n_estimators=tree_num, n_jobs=-1,criterion='entropy')
clf.fit(tr_x,tr_y)

print 'predicting at', datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

predict_y = clf.predict(val_x)

print 'Eval:',1.0*[predict_y[p] == val_y[p] for p in range(len(val_y))].count(False)/len(val_y)

print 'finish at', datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
