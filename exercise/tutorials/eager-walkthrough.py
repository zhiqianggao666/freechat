from __future__ import absolute_import,print_function,division

import  os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import pandas as pd

tf.enable_eager_execution()
print('Tensorflow version:{}'.format(tf.VERSION))
print('Eager execution:{}'.format(tf.executing_eagerly()))
train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
train_dataset_fp=tf.keras.utils.get_file('iris_training.csv',origin=train_dataset_url)
print('file:{}'.format(train_dataset_fp))
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
label_name=column_names[-1]
feature_names=column_names[:-1]
print('feature:{}'.format(feature_names))
print('label:{}'.format(label_name))
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
batchsize=32
train_dataset=tf.contrib.data.make_csv_dataset(train_dataset_fp,batchsize,column_names=column_names,label_name=label_name,num_epochs=1)
features,labels=next(iter(train_dataset))
print(features)
#print(labels)

plt.scatter(features['petal_length'],features['sepal_length'],c=labels,cmap='viridis')
plt.xlabel('Petal Length')
plt.ylabel('Sepal Length')
#plt.show()
a=features.values()
print('aaaaaa:', a)
b=list(a)
print('bbbbbb:', b)
c=tf.stack(b,axis=1)
print('cccccc:', c)

def pack_features_vector(features,labels):
    features=tf.stack(list(features.values()),axis=1)
    return features,labels

train_dataset=train_dataset.map(pack_features_vector)
features,labels=next(iter(train_dataset))
print(features[:5])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu,input_shape=(4,)),
    tf.keras.layers.Dense(10,activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
])
predictions=model(features)
print('predictions==================')
print(predictions[:5])
print(tf.nn.softmax(predictions[:5]))

print('predictions: {}'.format(tf.argmax(predictions,axis=1)))
print('predictions: {}'.format(tf.argmax(tf.nn.softmax(predictions[:5]),axis=1)))

print('Labels:{}'.format(labels))

def loss(model,x,y):
    y_=model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_)

l=loss(model,features,labels)
print(l)
def grad(model,inputs,targets):
    with tf.GradientTape() as tape:
        loss_value=loss(model,inputs,targets)
    return loss_value, tape.gradient(loss_value,model.trainable_variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
global_steps=tf.train.get_or_create_global_step()
loss_value,grads=grad(model,features,labels)
print('step:{},initial loss:{}'.format(global_steps.numpy(),loss_value.numpy()))
optimizer.apply_gradients(zip(grads,model.variables),global_steps)
print('Step:{}, loss:{}'.format(global_steps.numpy(),loss(model,features,labels).numpy()))

train_loss_results=[]
train_accuracy_results=[]
num_epochs=201
for epoch in range(num_epochs):
    epoch_loss_avg=tfe.metrics.Mean()
    epoch_accuracy=tfe.metrics.Accuracy()
    for x,y in train_dataset:
        loss_value,grads=grad(model,x,y)
        optimizer.apply_gradients(zip(grads,model.variables),global_steps)
        epoch_loss_avg(loss_value)
        epoch_accuracy(tf.argmax(model(x),axis=1,output_type=tf.int32),y)
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())
    if epoch%50==0:
        print('Epoch {:03d}: Loss{:.3f}, Accuracy:{:.3%}'.format(epoch,epoch_loss_avg.result(),epoch_accuracy.result()))


fig,axes=plt.subplots(2,sharex=True,figsize=(12,8))
fig.suptitle('Training Merics')
axes[0].set_ylabel('Loss', fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel('Accuracy',fontsize=14)
axes[1].set_xlabel('Epoch',fontsize=14)
axes[1].plot(train_accuracy_results)
#plt.show()

test_url = "http://download.tensorflow.org/data/iris_test.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename('iris_test.csv'),
                                  origin=test_url)

test_dataset=tf.contrib.data.make_csv_dataset(test_fp,
                                              batchsize,
                                              column_names=column_names,
                                              label_name='species',
                                              num_epochs=1,
                                              shuffle=False)
test_dataset=test_dataset.map(pack_features_vector)
test_accuracy=tfe.metrics.Accuracy()
for(x,y) in test_dataset:
    logits=model(x)
    predictions=tf.argmax(logits,axis=1,output_type=tf.int32)
    test_accuracy(predictions,y)
print('Test set accuracy: {:.3%}'.format(test_accuracy.result()))

print(tf.stack([y,predictions],axis=1))

predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5,],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]
])
predictions=model(predict_dataset)
for i,logits in enumerate(predictions):
    class_idx=tf.argmax(logits).numpy()
    p=tf.nn.softmax(logits)[class_idx]
    name=class_names[class_idx]
    print('example{} prediction:{} ({:4.1f}%)'.format(i, name, 100*p))


a=1