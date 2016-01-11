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
train_file = 'train_m.csv'
test_file = 'test_m.csv'
val_size = 0#19287
max_depth = 21
num_round = 1000#450
eta = 0.05
features_removed = []
train_or_load = 'train'
add_drop_rate_feature = False
track = 2

def drop_rate_statistics(x_y):
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
	#if track == 2:
		#for n in range(len(x_y)): x_y[n][-1] = x_y[n][-1]/np.sqrt(x_y[n][3])
		#for n in range(len(x_y)): x_y[n][-1] = (2.0*x_y[n][-1]-1)/np.sqrt(x_y[n][3])

	if add_drop_rate_feature: x_y = add_drop_rate_feature(x_y)

	tr_x = x_y[val_size:][:,:-1]
	tr_y = x_y[val_size:][:,-1]
	val_x = x_y[:val_size][:,:-1]
	val_y = x_y[:val_size][:,-1]

	dtrain = xgb.DMatrix(tr_x,label=tr_y)
	dtest = xgb.DMatrix(val_x,label=val_y)
	return dtrain,dtest


with open(dpath+test_file) as test:
	feature_list = test.readline().replace('\n','').split(',')
selected_features = range(len(feature_list))
for f in features_removed: selected_features.remove(f)

enrollment_dict_train = csv.DictReader(open(dpath+'enrollment_train.csv'))
enrollment_dict_test = csv.DictReader(open(dpath+'enrollment_test.csv'))
global enrollment_to_course
enrollment_to_course = {}
for e in enrollment_dict_train: enrollment_to_course[float(e['enrollment_id'])] = e['course_id']
for e in enrollment_dict_test: enrollment_to_course[float(e['enrollment_id'])] = e['course_id']

if train_or_load == 'train':
	with open('featmap.txt','w') as featmap:
		for s in range(len(selected_features)):
			featmap.write('%d\t%s\ti\n' %(s,feature_list[s]))
		featmap.write('%d\tdrop_count\tq\n' %len(feature_list))

	data = np.loadtxt(dpath+train_file, delimiter=',', skiprows=1,usecols=selected_features)
	label = np.loadtxt(dpath+'truth_train.csv', delimiter=',')
	dtrain,dval = draw(data,label,val_size)

	if track == 2:
		#objective = 'reg:linear'
		objective = 'binary:logistic'
		eval_metric = 'rmse'
	else:
		objective = 'binary:logistic'
		eval_metric = 'map@9000'

	param = {'max_depth':max_depth, 'eta':eta, 'silent':1, 'objective':objective,'booster':'gbtree',
	'nthread':multiprocessing.cpu_count(),'eval_metric':eval_metric}

	print 'max_depth:',max_depth,'num_round:',num_round,'eta:',eta

	if val_size > 0: watchlist  = [(dval,'eval'), (dtrain,'train')]
	else: watchlist  = [(dtrain,'train')]

	bst = xgb.train(param, dtrain, num_round, watchlist)

	if val_size > 0:
		preds = bst.predict(dval)
		labels = dval.get_label()
		error_type = 'val'
	elif val_size == 0:
		preds = bst.predict(dtrain)
		labels = dtrain.get_label()
		error_type = 'in'
	if track == 1: print 'E%s=%f' % ( error_type,sum([1 if int(preds[i]>=0.5)!=labels[i] else 0 for i in range(len(preds))]) /float(len(preds)) )
	#elif track == 2: print 'E%s=%f' % ( error_type,sum([1 if int(preds[i]>=0)!=labels[i] else 0 for i in range(len(preds))]) /float(len(preds)) )
	elif track == 2: print 'E%s=%f' % ( error_type,sum([1 if int(preds[i]>=0.5)!=labels[i] else 0 for i in range(len(preds))]) /float(len(preds)) )

	bst.save_model('xgb.model')
	bst.dump_model('dump.raw.txt')
	bst.dump_model('dump.nice.txt','./featmap.txt')
else:
	bst = xgb.Booster(model_file='xgb.model')

if val_size == 0:
	test_x = np.loadtxt(dpath+test_file, delimiter=',', skiprows=1,usecols=selected_features)
	if add_drop_rate_feature: test_x = add_drop_rate_feature(test_x,test=True)

	dtest = xgb.DMatrix(test_x)
	predict_test_y = bst.predict(dtest)
	if track == 1:
		result = sorted([[predict_test_y[t],int(test_x[t][0]),
			int((1+np.sign(predict_test_y[t]-0.5))/2)] for t in range(len(test_x))],reverse=True)
	elif track == 2:
		result = sorted([[predict_test_y[t],int(test_x[t][0]),
			int(predict_test_y[t]>=0.5)] for t in range(len(test_x))],reverse=True)
		#result = sorted([[(predict_test_y[t]*np.sqrt(test_x[t][3])+1)/2,int(test_x[t][0]),
			#int(np.sign(predict_test_y[t]))] for t in range(len(test_x))],reverse=True)

	submission = open('Submission_track%s.csv'%(track),'w')
	for r in result: submission.write('%s,%s\n' %(r[1],r[2]))
