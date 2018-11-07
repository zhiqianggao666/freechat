import  tensorflow  as tf
minist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test)=minist.load_data(path='mnist.zip')
x_train,x_test=x_train/255.0,x_test/255.0
x_train=x_train.reshape((len(x_train),784))
y_train=y_train.reshape((len(y_train),1))
BATCH_SIZE=32

train_dataset=tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset=train_dataset.map(lambda x,y:(tf.cast(x,dtype=tf.float32),tf.cast(y,dtype=tf.int64)))
train_dataset=train_dataset.batch(BATCH_SIZE).prefetch(2).repeat()
train_iterator=train_dataset.make_one_shot_iterator()
train_features,train_labels=train_iterator.get_next()
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512,activation=tf.nn.relu),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(10,activation=tf.nn.softmax)])

model.compile(optimizer=tf.train.AdamOptimizer(),loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])
model.fit(train_dataset,epochs=5,steps_per_epoch=int(len(y_train)/BATCH_SIZE))
#model.evaluate(x_test,y_test)

