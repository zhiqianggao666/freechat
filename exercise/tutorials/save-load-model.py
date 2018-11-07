from __future__ import absolute_import,division,print_function
import  tensorflow  as tf
from tensorflow import keras
import  os

minist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels)=minist.load_data(path='mnist.zip')
train_labels=train_labels[:10000]
test_labels=test_labels[:1000]

print(train_images[0].shape)
train_images=train_images[:10000].reshape(-1,28*28)/255.0
test_images=test_images[:1000].reshape(-1,28*28)/255.0

print(train_images[0].shape)

def create_model():
    model = tf.keras.Sequential()
    model.add(keras.layers.Dense(512,activation=tf.nn.relu,input_shape=(784,)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
    return model

checkpoint_path='training_1/cp.ckpt'
checkpoint_dir=os.path.dirname(checkpoint_path)
cp_callback=tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1)
model = create_model()
model.summary()
model.fit(train_images,train_labels,epochs=10,validation_data=(test_images,test_labels),callbacks=[cp_callback])

model = create_model()
model.load_weights(checkpoint_path)
loss,acc=model.evaluate(test_images,test_labels)
print('Restored model,accuracy:{:5.2f}%'.format(100*acc))

checkpoint_path='training_2/cp-{epoch:04d}.cpkt'
checkpoint_dir=os.path.dirname(checkpoint_path)
cp_callback=tf.keras.callbacks.ModelCheckpoint(checkpoint_path,verbose=1,save_weights_only=True,period=5)
model=create_model()
model.fit(train_images,train_labels,epochs=10,callbacks=[cp_callback],validation_data=(test_images,test_labels),verbose=0)

model.save_weights('./checkpoints/my_checkpoint')
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')
loss,acc=model.evaluate(test_images,test_labels)
print('Restored model, accuracy:{:5.2f}%'.format(100*acc))

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

model = create_model()
model.load_weights(latest)
loss,acc=model.evaluate(test_images,test_labels)
print('Restored model, accuracy:{:5.2f}%'.format(100*acc))


model = create_model()
model.fit(train_images,train_labels,epochs=5)
model.save('my_model.h5')
new_model = keras.models.load_model('my_model.h5')
#if use tf.train.optimizer, then need recompile again after loading.
#new_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.sparse_categorical_crossentropy,    metrics=['accuracy'])

new_model.summary()
loss,acc=new_model.evaluate(test_images,test_labels)
print('New Restored model, accuracy:{:5.2f}%'.format(100*acc))