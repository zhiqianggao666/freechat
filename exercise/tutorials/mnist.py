import  tensorflow  as tf
minist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test)=minist.load_data(path='mnist.zip')
x_train,x_test=x_train/255.0,x_test/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512,activation=tf.nn.relu),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(10,activation=tf.nn.softmax)])

model.compile(optimizer=tf.train.AdamOptimizer(),loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5)
#model.evaluate(x_test,y_test)

