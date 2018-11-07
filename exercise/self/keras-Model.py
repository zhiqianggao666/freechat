import tensorflow as tf
import numpy as np
BATCH_SIZE=128
minist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test)=minist.load_data(path='mnist.zip')
x_train,x_test=x_train/255.0,x_test/255.0

x_train=x_train.reshape((len(x_train),784))
y_train=y_train.reshape((len(y_train),1))
train_dataset=tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset=train_dataset.map(lambda x,y:(tf.cast(x,dtype=tf.float32),tf.cast(y,dtype=tf.int32)))
train_dataset=train_dataset.batch(BATCH_SIZE).repeat()
train_iterator=train_dataset.make_one_shot_iterator()
train_features,train_labels=train_iterator.get_next()

x_test=x_test.reshape((len(x_test),784))
y_test=y_test.reshape((len(y_test),1))
val_dataset=tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset=val_dataset.map(lambda x,y:(tf.cast(x,dtype=tf.float32),tf.cast(y,dtype=tf.int32)))
val_dataset=val_dataset.batch(BATCH_SIZE).repeat()
test_iterator=val_dataset.make_one_shot_iterator()
test_features,test_labels=test_iterator.get_next()


print('11111111111111111111')
inputs = tf.keras.Input(shape=(784,))
#x=tf.layers.flatten(inputs)
x=tf.layers.dense(inputs,512,activation=tf.nn.relu)
x=tf.layers.dropout(x,0.2)
predictions=tf.layers.dense(x,10,activation=tf.nn.softmax)
model = tf.keras.Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=tf.train.AdamOptimizer(),loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])
model.fit(train_features,train_labels,epochs=1,steps_per_epoch=int(len(y_train)/BATCH_SIZE))
#model.evaluate(test_features,test_labels,steps=int(len(y_test)/BATCH_SIZE))
print(model.summary())
def print_variable_count():
    total=0
    for variable in tf.trainable_variables():
        variable_p=1
        for dim in variable.shape:
            variable_p*=dim
        total+=variable_p
    print('total trainable variables: {}, size: {:.4f}M'.format(total,int(total)*4/(1024*1024)))

print_variable_count()
#tf.keras.estimator.model_to_estimator(model)
#print(model.sample_weights)
