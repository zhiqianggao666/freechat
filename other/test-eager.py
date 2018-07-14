from __future__ import absolute_import, division,print_function

import  os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()
#print(tf.VERSION)
#print(tf.executing_eagerly())

def split_func(line):
	a2=tf.compat.as_str(line)
	tf.string_strip()
	print(a2)
	a1=tf.decode_base64(line)
	a=line.numpy()
	b=a.decode()
	c= tf.string_split([line],'').values
	return c
	
dataset = tf.data.TextLineDataset(["/home/tizen/share/charmpy/middle-chat/Data/Corpus/Augment0/cornell_cleaned_new.txt"])
dataset = dataset.map(split_func)

iter = tfe.Iterator(dataset)
a = iter.next()
print(iter.next().numpy().decode('UTF-8'))

while(1):
	data = iter.next().numpy()
	if(data):
		print(data.decode('UTF-8')[0])
		print(data.decode('UTF-8')[1])
		print(data.decode('UTF-8')[2])
		print(data.decode('UTF-8')[3])
		print(data.decode('UTF-8')[4])
		print(data.decode('UTF-8')[5])
		print(data.decode('UTF-8').split())
	else:
		print('end-------')
		break



b2 = tfe.Variable([[1,2],[3,4]], name='b')
print (b2)


train_dataset_url="http://download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),origin=train_dataset_url)
#print(train_dataset_fp)

def parse_csv(line):
	example_defaults=[[0.],[0.],[0.],[0.],[0]]
	parsed_line=tf.decode_csv(line,example_defaults)
	
	
	features=tf.reshape(parsed_line[:-1],shape=[4,])
	label=tf.reshape(parsed_line[-1],shape=())
	#print(features[0])

	return features,label

train_dataset=tf.data.TextLineDataset(train_dataset_fp)
#print(train_dataset)
train_dataset=train_dataset.skip(1)
train_dataset=train_dataset.map(parse_csv)
train_dataset=train_dataset.shuffle(buffer_size=1000)
train_dataset=train_dataset.batch(32)

it = train_dataset.make_one_shot_iterator()
print(it.next())

features,label=iter(train_dataset).next()

print(features[0])
print(label[0])
model=tf.keras.Sequential([tf.keras.layers.Dense(10,activation='relu',input_shape=(4,)),
                           tf.keras.layers.Dense(10,activation='relu'),
                           tf.keras.layers.Dense(3)])
def loss(model,x,y):
	y_=model(x)
	return tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_)

def grad(model,inputs,targets):
	with tf.GradientTape() as tape:
		loss_value = loss(model,inputs,targets)
	return tape.gradient(loss_value,model.variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

train_loss_results=[]
train_accuracy_results=[]

num_epochs=201
for epoch in range(num_epochs):
	epoch_loss_avg=tfe.metrics.Mean()
	epoch_accuracy=tfe.metrics.Accuracy()
	
	for x,y in train_dataset:
		grads=grad(model,x,y)
		optimizer.apply_gradients(zip(grads,model.variables),global_step=tf.train.get_or_create_global_step())
		
		epoch_loss_avg(loss(model,x,y))
		epoch_accuracy(tf.argmax(model(x),axis=1,output_type=tf.int32),y)
	train_loss_results.append(epoch_loss_avg.result())
	train_accuracy_results.append(epoch_accuracy.result())
	
	if epoch % 50 == 0:
		print('Epoch{}:Loss:{},Accuracy:{}'.format(epoch, epoch_loss_avg.result(),epoch_accuracy.result()))
		


fig,axes=plt.subplots(2,sharex=True,figsize=(12,8))
fig.suptitle('training metrics')
axes[0].set_ylabel('Loss', fontsize=14)
#axes[0].set_xlabel('Epoch',fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel('Accuracy',fontsize=14)
axes[1].set_xlabel('Epoch',fontsize=14)
axes[1].plot(train_accuracy_results)

plt.show()



test_url='http://download.tensorflow.org/data/iris_test.csv'
test_fp=tf.keras.utils.get_file(fname=os.path.basename(test_url),origin=test_url)
test_dataset=tf.data.TextLineDataset(test_fp)
test_dataset=test_dataset.skip(1)
test_dataset=test_dataset.map(parse_csv)
test_dataset=test_dataset.shuffle(1000)
test_dataset=test_dataset.batch(32)

test_accuracy=tfe.metrics.Accuracy()
for(x,y) in test_dataset:
	prediction=tf.argmax(model(x),axis=1,output_type=tf.int32)
	test_accuracy(prediction, y)
	
print('Test set accuracy:{}'.format(test_accuracy.result()))

class_ids=['Iris setosa', 'Iris versicolor','Iris virginica']
predict_dataset=tf.convert_to_tensor([
	[5.1,3.3,1.7,0.5],
	[5.9,3.0,4.2,1.5],
	[6.9,3.1,5.4,2.1]
])

predictions=model(predict_dataset)

for i,logits in enumerate(predictions):
	class_idx=tf.argmax(logits).numpy()
	name=class_ids[class_idx]
	print('example:{},prediction:{}'.format(i,name))



a=1