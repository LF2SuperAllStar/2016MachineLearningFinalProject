import numpy as np
import scipy.sparse
import pickle
import xgboost as xgb
import random
import multiprocessing
import os
import csv

if os.path.isdir("/tmp2/r03921017/data/"): dpath = '/tmp2/r03921017/data/'
else: dpath = '../data/'
train_file = 'train_k.csv'
test_file = 'test_k.csv'
val_size = 19287
max_depth = 22
num_round = 100#300
eta = 0.15#0.05
features_removed = [19]
train_or_load = 'train'

def drop_rate_statistics(x_y):
	drop_list = csv.reader(open(dpath+'truth_train.csv'))
	enrollment_to_drop = {}
	for drop in drop_list:
		enrollment_to_drop[float(drop[0])] = int(drop[1])

	course_drop_rate = {}
	for enrollment in x_y[val_size:]:
		enrollment_id = enrollment[0]
		course = enrollment_to_course[enrollment_id]
		drop = enrollment_to_drop[enrollment_id]
		if course not in course_drop_rate: course_drop_rate[course] = [drop,1]
		else:
			course_drop_rate[course][0] += drop
			course_drop_rate[course][1] += 1

	for course in course_drop_rate:
		course_drop_rate[course] = 1.0*course_drop_rate[course][0]/course_drop_rate[course][1]
	return course_drop_rate

def add_drop_rate_feature(x_y):
	global enrollment_to_course
	enrollment_dict = csv.DictReader(open(dpath+'enrollment_train.csv'))
	enrollment_to_course = {}
	for e in enrollment_dict: enrollment_to_course[float(e['enrollment_id'])] = e['course_id']
	course_drop_rate = drop_rate_statistics(x_y)
	drop_rate_list = np.array([[course_drop_rate[enrollment_to_course[enrollment[0]]]] for enrollment in x_y[val_size:]])
	return np.append(x_y[val_size:][:,:-1],drop_rate_list,axis=1)

def draw(data,label,val_size):
	x_y = np.array([np.append(data[n],label[n][-1]) for n in range(len(data))])
	random.shuffle(x_y)

	#tr_x = add_drop_rate_feature(x_y)
	tr_x = x_y[val_size:][:,:-1]
	tr_y = x_y[val_size:][:,-1]
	val_x = x_y[:val_size][:,:-1]
	val_y = x_y[:val_size][:,-1]
	dtrain = xgb.DMatrix(tr_x,label=tr_y)
	dtest = xgb.DMatrix(val_x,label=val_y)
	return dtrain,dtest

if train_or_load == 'train':
	with open(dpath+test_file) as test:
		feature_list = test.readline().split(',')
	selected_features = range(len(feature_list))
	for f in features_removed: selected_features.remove(f)
	featmap = open('featmap.txt','w')
	for s in range(len(selected_features)):
		featmap.write('%s\t%s\ti\n' %(s,feature_list[s]))

	data = np.loadtxt(dpath+train_file, delimiter=',', skiprows=1,usecols=selected_features)
	label = np.loadtxt(dpath+'truth_train.csv', delimiter=',')
	dtrain,dval = draw(data,label,val_size)

	# specify parameters via map, definition are same as c++ version
	param = {'max_depth':max_depth, 'eta':eta, 'silent':1, 'objective':'binary:logistic','booster':'gbtree',
	'nthread':multiprocessing.cpu_count(),'eval_metric':'map'}#@9000'}

	print 'max_depth:',max_depth,'num_round:',num_round,'eta:',eta

	# specify validations set to watch performance
	if val_size > 0: watchlist  = [(dval,'eval'), (dtrain,'train')]
	else: watchlist  = [(dtrain,'train')]

	bst = xgb.train(param, dtrain, num_round, watchlist)

	if val_size > 0:
		preds = bst.predict(dval)
		labels = dval.get_label()
		print ('Eval=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))
	else:
		preds = bst.predict(dtrain)
		labels = dtrain.get_label()
		print ('Ein=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))

	bst.save_model('xgb.model')
	# dump model
	bst.dump_model('dump.raw.txt')
	# dump model with feature map
	bst.dump_model('dump.nice.txt','./featmap.txt')
else:
	bst = xgb.Booster(model_file='xgb.model')



test_x = np.loadtxt(dpath+test_file, delimiter=',', skiprows=1,usecols=selected_features)

dtest = xgb.DMatrix(test_x)
preds_test = bst.predict(dtest)

result = sorted([[preds_test[t],int(test_x[t][0]),int((1+np.sign(preds_test[t]-0.5))/2)] for t in range(len(test_x))],reverse=True)

submission = open('Submission.csv','w')
for r in result: submission.write('%s,%s\n' %(r[1],r[2]))
