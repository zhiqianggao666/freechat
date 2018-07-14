from preprocessing import *
from featureEncoding import *
from svm import svm_problem, svm_parameter
from svmutil import svm_train, svm_predict, svm_save_model, svm_read_problem, svm_load_model
import argparse
from sklearn.utils import shuffle
from sklearn import preprocessing # preprocessing.scale(X)
from sklearn.metrics import confusion_matrix

TRAIN_SET_RATIO = 1.0
PATCH_SIZE = 20
def test_predict(filelist):
	model_filelist = ['./dataset1/in_ts.csv', './dataset1/in_wsq.csv', './dataset1/in_wsw.csv', './dataset1/in_xyj.csv', './dataset1/in_lzq.csv', './dataset1/in_zyj.csv', './dataset1/in_ykr.csv', './dataset1/in_zxf.csv']
	#filelist = ['./dataset1/in_ts.csv', './dataset1/in_wsq.csv', './dataset1/in_wsw.csv', './dataset1/in_xyj.csv', './dataset1/in_lzq.csv', './dataset1/in_zyj.csv', './dataset1/in_ykr.csv', './dataset1/in_zxf.csv', './dataset1/in_lzq2.csv', './dataset1/in_zyj2.csv', './dataset1/in_wsw2.csv', './dataset1/in_xyj2.csv', './dataset1/in_wsq2.csv', './dataset1/in_zxf2.csv']
	#filelist = ['./dataset1/in_lzq2.csv', './dataset1/in_zyj2.csv', './dataset1/in_wsw2.csv', './dataset1/in_xyj2.csv', './dataset1/in_wsq2.csv', './dataset1/in_zxf2.csv']
	#filelist0 = ['./dataset1/in_zxf2.csv']
	#filelist1 = ['./dataset1/in_zyj2.csv']
	#filelist2 = ['./dataset1/in_lzq2.csv']
	#filelist3 = ['./dataset1/in_wsw2.csv']
	#filelist4 = ['./dataset1/in_wsq2.csv']
	#filelist5 = ['./dataset1/in_xyj2.csv']
	display_name = ['Tang Shuai','Wang Shuqing', 'Wang Songwen', 'Xiao Yujie', 'Liang Zuqin', 'Zhou Yuanjia', 'Yin Kairong', 'Zhang Xiaofei']
	argparser = argparse.ArgumentParser()
	#argparser.add_argument('fileName', type=str, help='fileName')
	args = argparser.parse_args()
	args = vars(args)
	dataSet = []
	label = []
	new_index = 0
	filearr = []
	filearr.append(filelist)
	for i, fileName in enumerate( filearr ):
		tmp = readDataset(fileName) # array of Instance
		dataSet = dataSet + tmp
		print ('size:', len(tmp))
		label = label + [i]*len(tmp)
		dataSet, label = shuffle(dataSet, label, random_state=0)
		
	cutIndex = int(TRAIN_SET_RATIO*len(dataSet))
	## use accel_abs and alpha_abs as input for encoding respectively
	print ('learning dictionary...')
	data_accel = [I.accel_abs() for I in dataSet]
	data_alpha = [I.alpha_abs() for I in dataSet]
	RPDictionary_accel = Dictionary1(PATCH_SIZE, 0, data_accel[:cutIndex])
	RPDictionary_alpha = Dictionary1(PATCH_SIZE, 1, data_alpha[:cutIndex])
	print ('feature encoding...')
	aggregate_feature = [ f[0]+f[1] for f in zip( RPDictionary_accel.encoding(data_accel), RPDictionary_alpha.encoding(data_alpha) ) ]
	#aggregate_feature = preprocessing.scale(aggregate_feature) ## scale columns independently to have zero mean and unit variance
	
	print (len(aggregate_feature[:cutIndex]),len(aggregate_feature[cutIndex:]))
	writeFeature('./svm_test_run', aggregate_feature[:cutIndex], label[:cutIndex])

	print ('SVM predicting...')
	ballot_box = []
	X_test, Y_test = readFeaturePos('./svm_test_run',PATCH_SIZE*2, new_index)
	max_acc=0.0
	max_index=0
	for i, fileName in enumerate( model_filelist):
		Y_test_i = [int(y==new_index) for y in Y_test]
		model = svm_load_model('{}.model'.format(i))
		p_labels, p_acc, p_vals = svm_predict(Y_test_i, X_test, model)
		#ballot_box.append(p_labels)
		#print p_acc
		if max_acc < p_acc[0]:
			max_acc = p_acc[0]
			max_index = i
			
	print ('The gait data belongs to \'' , display_name[max_index] , '\', accuracy is:' , max_acc)
	return max_index+1,max_acc
	#ballot_box = np.array(ballot_box)
	#Y_predict = [(sum(ballot_box[:,i])==0) for i in range(len(Y_test))]
	#Y_test = [int(y==4) for y in Y_test]
	# true\predicted
	#print confusion_matrix(Y_test, Y_predict)
	
	
if __name__=='__main__':
	model_filelist = ['./dataset1/in_ts.csv', './dataset1/in_wsq.csv', './dataset1/in_wsw.csv', './dataset1/in_xyj.csv', './dataset1/in_lzq.csv', './dataset1/in_zyj.csv', './dataset1/in_ykr.csv', './dataset1/in_zxf.csv']
	#filelist = ['./dataset1/in_ts.csv', './dataset1/in_wsq.csv', './dataset1/in_wsw.csv', './dataset1/in_xyj.csv', './dataset1/in_lzq.csv', './dataset1/in_zyj.csv', './dataset1/in_ykr.csv', './dataset1/in_zxf.csv', './dataset1/in_lzq2.csv', './dataset1/in_zyj2.csv', './dataset1/in_wsw2.csv', './dataset1/in_xyj2.csv', './dataset1/in_wsq2.csv', './dataset1/in_zxf2.csv']
	#filelist = ['./dataset1/in_lzq2.csv', './dataset1/in_zyj2.csv', './dataset1/in_wsw2.csv', './dataset1/in_xyj2.csv', './dataset1/in_wsq2.csv', './dataset1/in_zxf2.csv']
	filelist0 = ['./dataset1/in_zxf2.csv']
	filelist1 = ['./dataset1/in_zyj2.csv']
	filelist2 = ['./dataset1/in_lzq2.csv']
	filelist3 = ['./dataset1/in_wsw2.csv']
	filelist4 = ['./dataset1/in_wsq2.csv']
	filelist5 = ['./dataset1/in_xyj2.csv']
	display_name = ['Tang Shuai','Wang Shuqing', 'Wang Songwen', 'Xiao Yujie', 'Liang Zuqin', 'Zhou Yuanjia', 'Yin Kairong', 'Zhang Xiaofei']
	argparser = argparse.ArgumentParser()
	#argparser.add_argument('fileName', type=str, help='fileName')
	args = argparser.parse_args()
	args = vars(args)
	dataSet = []
	label = []
	new_index = 0
	for i, fileName in enumerate( filelist5 ):
		tmp = readDataset(fileName) # array of Instance
		dataSet = dataSet + tmp
		print ('size:', len(tmp))
		label = label + [i]*len(tmp)
		dataSet, label = shuffle(dataSet, label, random_state=0)
		
	cutIndex = int(TRAIN_SET_RATIO*len(dataSet))
	## use accel_abs and alpha_abs as input for encoding respectively
	print ('learning dictionary...')
	data_accel = [I.accel_abs() for I in dataSet]
	data_alpha = [I.alpha_abs() for I in dataSet]
	RPDictionary_accel = Dictionary1(PATCH_SIZE, 0, data_accel[:cutIndex])
	RPDictionary_alpha = Dictionary1(PATCH_SIZE, 1, data_alpha[:cutIndex])
	print ('feature encoding...')
	aggregate_feature = [ f[0]+f[1] for f in zip( RPDictionary_accel.encoding(data_accel), RPDictionary_alpha.encoding(data_alpha) ) ]
	#aggregate_feature = preprocessing.scale(aggregate_feature) ## scale columns independently to have zero mean and unit variance
	
	print (len(aggregate_feature[:cutIndex]),len(aggregate_feature[cutIndex:]))
	writeFeature('./svm_test_run', aggregate_feature[:cutIndex], label[:cutIndex])

	print ('SVM predicting...')
	ballot_box = []
	X_test, Y_test = readFeaturePos('./svm_test_run',PATCH_SIZE*2, new_index)
	max_acc=0.0
	max_index=0
	for i, fileName in enumerate( model_filelist):
		Y_test_i = [int(y==new_index) for y in Y_test]
		model = svm_load_model('{}.model'.format(i))
		p_labels, p_acc, p_vals = svm_predict(Y_test_i, X_test, model)
		#ballot_box.append(p_labels)
		#print p_acc
		if max_acc < p_acc[0]:
			max_acc = p_acc[0]
			max_index = i
			
	print ('The gait data belongs to \'' , display_name[max_index] , '\', accuracy is:' , max_acc)
	
	#ballot_box = np.array(ballot_box)
	#Y_predict = [(sum(ballot_box[:,i])==0) for i in range(len(Y_test))]
	#Y_test = [int(y==4) for y in Y_test]
	# true\predicted
	#print confusion_matrix(Y_test, Y_predict)
