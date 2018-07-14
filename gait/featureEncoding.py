import math
import random
import numpy as np
from preprocessing import *

def dynamicTimeWarp(seqA, seqB, d = lambda x,y: abs(x-y), print_flag = False):
	# create the cost matrix
	numRows, numCols = len(seqA), len(seqB)
	cost = [[0 for _ in range(numCols)] for _ in range(numRows)]

	# initialize the first row and column
	cost[0][0] = d(seqA[0], seqB[0])
	for i in range(1, numRows):
		cost[i][0] = cost[i-1][0] + d(seqA[i], seqB[0])

	for j in range(1, numCols):
		cost[0][j] = cost[0][j-1] + d(seqA[0], seqB[j])
 
	# fill in the rest of the matrix
	for i in range(1, numRows):
		for j in range(1, numCols):
			choices = cost[i-1][j], cost[i][j-1], cost[i-1][j-1]
			cost[i][j] = min(choices) + d(seqA[i], seqB[j])
	if print_flag:
		for row in cost:
			for entry in row:
				print ("%03d" % entry,)
			print ("")
	return cost

def dtwDistanceMatrix(instances, metric='dtw',down_sample = True):
    # O(n^3), down sampling is imperative
    DOWN_SAMPLING_RATIO = 10
    l = len(instances)
    w = len(instances[0])
    if down_sample:
        instances_low = np.zeros((l,w//DOWN_SAMPLING_RATIO))
        for i in range(l):
            instance = np.array(instances[i])
            instances_low[i] = instance[range(0,w,DOWN_SAMPLING_RATIO)]
        instances = instances_low
    print (instances.shape)
    d = np.ones((l,l)) * -1
    for i in range(l):
        for j in range(i+1):
            if 'dtw' in metric:
                distance = dynamicTimeWarp(instances[i], instances[j])[-1][-1]
            elif 'euclidean' in metric:
                distance = np.linalg.norm(instances[i] - instances[j], ord=10)
            else:
                print('No such metric!')
            d[i,j] = distance
            d[j,i] = distance
    return d
            

def writeFeature(fileName, instances, label=[]):
	# wtite features into libsvm format
	if len(label) == 0:
		label = ['-1']*len(instances)
	elif len(label) != len(instances):
		print ('ERROR : len(label) != len(instances)')
	with open(fileName, 'w') as fw:
		for j, features in enumerate(instances):
			feature_str = ''
			for i, f in enumerate(features):
				if f != 0:
					feature_str = ' '.join( [feature_str, ':'.join([str(i),str(f)])] )
			print >> fw, ' '.join([ str(int(label[j])), feature_str ])
	return

def readFeature(fileName, featureNum):
	features = []
	labels = []
	with open(fileName, 'r') as fr:
		for line in fr:
			line = line.split()
			instance = [0]*featureNum
			for f in line[1:]:
				i = int( f.split(':')[0] )
				instance[i] = float(f.split(':')[-1])
			features.append(instance)
			labels.append(int(line[0]))
	return features, labels	
	
def readFeaturePos(fileName, featureNum, pos):
	features = []
	labels = []
	with open(fileName, 'r') as fr:
		for line in fr:
			line = line.split()
			if int(line[0]) == pos:
				instance = [0]*featureNum
				for f in line[1:]:
					i = int( f.split(':')[0] )
					instance[i] = float(f.split(':')[-1])
				features.append(instance)
				labels.append(int(line[0]))
	return features, labels

class Dictionary1:
	def __init__(self, atomNum, flag, instances, distType = 'DTW'):
		# random patch dictionary
		# atomNum had better be much larger than the number of classes
		#self.atoms = random.sample(instances, atomNum)
		
		tmp = readDataset('./dataset1/in_ts.csv') # array of Instance
		self.atoms = []
		if flag == 0:
			dd = [I.accel_abs() for I in tmp]
		else:
			dd = [I.alpha_abs() for I in tmp]
		self.atoms = dd[8:12];
		self.atoms = self.atoms + dd[19:25];
		self.atoms = self.atoms + dd[29:33];
		self.atoms = self.atoms + dd[35:39];
		self.atoms = self.atoms + dd[40:42];
		self.distType = distType
	def encoding(self, instances):
		encodedInstances = []
		for i in instances:
			encodedInstances.append( [self.__dist(i, a) for a in self.atoms] )
		return encodedInstances
	def getAtoms(self):
		return self.atoms
	def __dist(self, seqA, seqB):
		if self.distType == 'DTW':
			return dynamicTimeWarp(seqA, seqB)[-1][-1]
		else:
			print ('distType ERROR!')
			return

			
class Dictionary:
	def __init__(self, atomNum, flag, instances, distType = 'DTW'):
		# random patch dictionary
		# atomNum had better be much larger than the number of classes
		self.atoms = random.sample(instances, atomNum)
		self.distType = distType
	def encoding(self, instances):
		encodedInstances = []
		for i in instances:
			encodedInstances.append( [self.__dist(i, a) for a in self.atoms] )
		return encodedInstances
	def getAtoms(self):
		return self.atoms
	def __dist(self, seqA, seqB):
		if self.distType == 'DTW':
			return dynamicTimeWarp(seqA, seqB)[-1][-1]
		else:
			print ('distType ERROR!')
			return
			
if __name__ == '__main__':
	seqA = [0, 0, 0, 3, 6, 13, 25, 22, 7, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	seqB = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 12, 24, 23, 8 ,3, 1, 0, 0, 0, 0, 0]
	print ('seqA', seqA)
	print ('seqB', seqB)
	print ('cost =', dynamicTimeWarp(seqA, seqB)[-1][-1])
	RPDictionary = Dictionary(2, [seqA, seqB])
	writeFeature('./deleteMe', RPDictionary.encoding([seqA, seqB]))
