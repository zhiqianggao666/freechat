import tensorflow as tf
from tensorflow import keras

model = keras.Sequential()
model.add(keras.layers.Dense(64,activation=tf.nn.relu))
model.add(keras.layers.Dense(64,kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(keras.layers.Dense(64,kernel_initializer=tf.initializers.orthogonal))
model.add(keras.layers.Dense(64,bias_initializer=tf.initializers.constant(2.0)))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss=keras.losses.categorical_crossentropy, metrics=[tf.keras.metrics.categorical_accuracy])

import numpy as np
data = np.random.random((1000,32))
labels=np.random.random((1000,10))

val_data=np.random.random((100,32))
val_labels=np.random.random((100,10))
model.fit(data, labels,epochs=10,batch_size=32, validation_data=(val_data,val_labels))

dataset = tf.data.Dataset.from_tensor_slices((data,labels))
dataset=dataset.batch(32)
dataset=dataset.repeat()

val_dataset=tf.data.Dataset.from_tensor_slices((val_data,val_labels))
val_dataset=val_dataset.batch(32).repeat()

model.fit(dataset,epochs=10,steps_per_epoch=30, validation_data=val_dataset,validation_steps=3)

model.evaluate(val_data,steps=3)
model.predict(val_data,steps=30)

input=keras.Input(shape=(32,))
x=keras.layers.Dense(64,activation=tf.nn.relu)(input)
x=keras.layers.Dense(64,activation=tf.nn.relu)(x)
predictions=keras.layers.Dense(10,activation=tf.nn.softmax)(x)
model=keras.Model(input=input,outputs=predictions)
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),loss=tf.keras.losses.categorical_crossentropy,metrics=[tf.keras.metrics.categorical_accuracy])
model.fit(data,labels,batch_size=32,epochs=5)

class MyModel(keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes= num_classes
        self.dense_1=keras.layers.Dense(32,activation=tf.nn.relu)
        self.dense_2=keras.layers.Dense(num_classes,activation=tf.nn.sigmoid)

    def call(self, inputs, training=None, mask=None):
        x=self.dense_1(inputs)
        return self.dense_2(x)
    def compute_output_shape(self, input_shape):
        shape=tf.TensorShape(input_shape).as_list()
        shape[-1]=self.num_classes
        return tf.TensorShape(shape)
model = MyModel(num_classes=10)
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),loss=tf.keras.losses.categorical_crossentropy,metrics=[tf.keras.metrics.categorical_accuracy])
model.fit(data,labels,batch_size=32,epochs=5)

class MyLayer(keras.layers.Layer):
    def __init__(self,output_dim,**kwargs):
        self.output_dim=output_dim
        super(MyLayer,self).__init__(**kwargs)

    def build(self, input_shape):
        shape=tf.TensorShape((input_shape[1],self.output_dim))
        self.kernel=self.add_weight(name='kernel',shape=shape,initializer='uniform',trainable=True)
        super(MyLayer,self).build(input_shape)
    def call(self, inputs, **kwargs):
        return tf.matmul(inputs,self.kernel)
    def compute_output_shape(self, input_shape):
        shape=tf.TensorShape(input_shape).as_list()
        shape[-1]=self.output_dim
        return tf.TensorShape(shape)
    def get_config(self):
        base_config=super(MyLayer,self).get_config()
        base_config['output_dim']=self.output_dim
        return base_config
    def from_config(cls, config):
        return cls(**config)

model = keras.Sequential(MyLayer(10),keras.layers.Activation('softmax'))
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001), loss=tf.keras.losses.categorical_crossentropy,metrics=[keras.metrics.categorical_accuracy])
model.fit(data,labels,batch_size=32,epochs=5)

callbacks=[keras.callbacks.EarlyStopping(patience=2,monitor='val_loss'),keras.callbacks.TensorBoard(log_dir='./logs')]
model.fit(data,labels,batch_size=32,epochs=5,callbacks=callbacks,validation_data=(val_data,val_labels))

model.save_weights('./my_model')
model.load_weights('my_model')

json_sting=model.to_json()
fresh_model=keras.models.model_from_json(json_sting)
yaml_string=model.to_yaml()
fresh_model=keras.models.model_from_yaml(yaml_string)

model=keras.Sequential([
    keras.layers.Dense(10,activation=tf.nn.softmax,input_shape=(32,)),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),loss=tf.keras.losses.categorical_crossentropy,metrics=[keras.metrics.categorical_accuracy])
model.fit(data,labels,batch_size=32,epochs=5)
model.save('my_model.h5')
model=keras.models.load_model('my_model.h5')

model = keras.Sequential([
    keras.layers.Dense(10,activation=tf.nn.softmax),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),loss=tf.keras.losses.categorical_crossentropy,metrics=[keras.metrics.categorical_accuracy])
estimator=keras.estimator.model_to_estimator(model)


model = keras.Sequential([
    keras.layers.Dense(10,activation=tf.nn.relu, input_shape=(10,)),
    keras.layers.Dense(10,activation=tf.nn.sigmoid)
])
optimizer=tf.train.GradientDescentOptimizer(0.2)
model.compile(optimizer=optimizer,loss=tf.keras.losses.binary_crossentropy)
model.summary()

def input_fn():
    x=np.random.random((1024,10))
    y=np.random.randint(2,size=(1024,1))
    x=tf.cast(x,tf.float32)
    dataset=tf.data.Dataset.from_tensor_slices((x,y))
    dataset=dataset.repeat(10)
    dataset=dataset.batch(32)
    return dataset
from tensorflow.contrib import distribute
strategy=distribute.MirroredStrategy()
config=tf.estimator.RunConfig(train_distribute=strategy)
keras_estimator=keras.estimator.model_to_estimator(keras_model=model,config=config,model_dir='/tmp/model_dir')
keras_estimator.train(input_fn=input_fn,steps=10)
