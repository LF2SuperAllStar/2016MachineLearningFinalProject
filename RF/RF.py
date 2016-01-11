from sklearn.ensemble import RandomForestClassifier
import csv
from multiprocessing import Pool
import numpy as np
import time
import datetime
import random
import os

if os.path.isdir("/tmp2/r03921017/data/"): dpath = '/tmp2/r03921017/data/'
else: dpath = '../data/'
train_file = 'train_m.csv'
test_file = 'test_m.csv'
val_size = 19287#0
tree_num  = 20000

def drop_rate_statistics(x_y):
	enrollment_dict_train = csv.DictReader(open(dpath+'enrollment_train.csv'))
	enrollment_dict_test = csv.DictReader(open(dpath+'enrollment_test.csv'))
	global enrollment_to_course
	enrollment_to_course = {}
	for e in enrollment_dict_train: enrollment_to_course[float(e['enrollment_id'])] = e['course_id']
	for e in enrollment_dict_test: enrollment_to_course[float(e['enrollment_id'])] = e['course_id']

	drop_list = csv.reader(open(dpath+'truth_train.csv'))
	enrollment_to_drop = {}
	for drop in drop_list:
		enrollment_to_drop[float(drop[0])] = int(drop[1])

	global course_drop_rate
	course_drop_rate = {}
	for enrollment in x_y:
		enrollment_id = enrollment[0]
		course = enrollment_to_course[enrollment_id]
		drop = enrollment_to_drop[enrollment_id]
		if course not in course_drop_rate: course_drop_rate[course] = [drop,1]
		else:
			course_drop_rate[course][0] += drop
			course_drop_rate[course][1] += 1

	with open('drop_count','w') as drop_count:
		drop_count.write('course,drop_count\n')
		for course in course_drop_rate:
			course_drop_rate[course] = 1.0*course_drop_rate[course][0]#/course_drop_rate[course][1]
			drop_count.write('%s,%s\n' %(course,course_drop_rate[course]))

def add_drop_rate_feature(x_y,test=False):
	if test:
		with open('drop_count') as drop_count:
			course_to_drop = {}
			next(drop_count)
			for line in drop_count: course_to_drop[line.split(',')[0]]=float(line.split(',')[1])
			drop_rate_list = np.array([[course_to_drop[enrollment_to_course[enrollment[0]]]] for enrollment in x_y])
			return np.append(x_y,drop_rate_list,axis=1)
	else:
		drop_rate_statistics(x_y[val_size:])
		drop_rate_list = np.array([[course_drop_rate[enrollment_to_course[enrollment[0]]]] for enrollment in x_y])
		added_x_y = []
		for n in range(len(x_y)): added_x_y.append(np.insert(x_y[n],-1,[drop_rate_list[n]]))
		return np.array(added_x_y)

def draw(data,label,val_size):
	x_y = np.array([np.append(data[n],label[n][-1]) for n in range(len(data))])
	random.shuffle(x_y)

	x_y = add_drop_rate_feature(x_y)

	tr_x = x_y[val_size:][:,:-1]
	tr_y = x_y[val_size:][:,-1]
	val_x = x_y[:val_size][:,:-1]
	val_y = x_y[:val_size][:,-1]
	return tr_x,tr_y,val_x,val_y

data = np.loadtxt(dpath+train_file, delimiter=',', skiprows=1)
label = np.loadtxt(dpath+'truth_train.csv', delimiter=',')
tr_x,tr_y,val_x,val_y = draw(data,label,val_size)

print 'training at ', datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

clf = RandomForestClassifier(n_estimators=tree_num,n_jobs=-1,criterion='entropy',warm_start=True)#,max_leaf_nodes=8000)
clf.fit(tr_x,tr_y)

print 'predicting at', datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
if val_size>0:
	predict_y = clf.predict(val_x)
	print 'Eval:',1.0*[predict_y[p] == val_y[p] for p in range(len(val_y))].count(False)/len(val_y)
else:
	test_x = np.loadtxt(dpath+test_file, delimiter=',', skiprows=1,usecols=selected_features)
	test_x = add_drop_rate_feature(test_x,test=True)

	predict_test_y = clf.predict(test_x)
	result = sorted([[predict_test_y[t],int(test_x[t][0]),int((1+np.sign(predict_test_y[t]-0.5))/2)] for t in range(len(test_x))],reverse=True)

	submission = open('Submission.csv','w')
	for r in result: submission.write('%s,%s\n' %(r[1],r[2]))