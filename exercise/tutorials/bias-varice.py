import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

imdb=keras.datasets.imdb
NUM_WORDS=10000
(train_data,train_labels),(test_data,test_labels)=imdb.load_data('imdb.zip',num_words=NUM_WORDS)
print('Training entries:{}, lables{}'.format(len(train_data),len(train_labels)))
print(train_data[0])
print(len(train_data[0]), len(train_data[1]))

def multi_hot_sequence(sequences, dimension):
    results = np.zeros((len(sequences),dimension))
    for i, word_indices in enumerate(sequences):
        results[i,word_indices]=1.0
    return results

train_data=multi_hot_sequence(train_data, dimension=NUM_WORDS)
test_data=multi_hot_sequence(test_data,dimension=NUM_WORDS)

plt.plot(train_data[0])
#plt.show()
baseline_model = keras.Sequential()
baseline_model.add(keras.layers.Dense(16, activation=tf.nn.relu,input_shape=(NUM_WORDS,)))
baseline_model.add(keras.layers.Dense(16,activation=tf.nn.relu))
baseline_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
baseline_model.compile(optimizer=tf.train.AdamOptimizer(),loss='binary_crossentropy',metrics=['accuracy','binary_crossentropy'])
baseline_model.summary()

baseline_history=baseline_model.fit(train_data,train_labels,epochs=20,batch_size=512,validation_data=(test_data,test_labels),verbose=2)

smaller_model=keras.Sequential()
smaller_model.add(tf.keras.layers.Dense(4, activation=tf.nn.relu,input_shape=(NUM_WORDS,)))
smaller_model.add(keras.layers.Dense(4,activation=tf.nn.relu))
smaller_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
smaller_model.compile(optimizer=tf.train.AdamOptimizer(),loss='binary_crossentropy',metrics=['accuracy','binary_crossentropy'])
smaller_model.summary()

smaller_history=smaller_model.fit(train_data,train_labels,epochs=20,batch_size=512,validation_data=(test_data,test_labels),verbose=2)


bigger_model = keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

bigger_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy','binary_crossentropy'])

bigger_model.summary()
# bigger_history = bigger_model.fit(train_data, train_labels,
#                                   epochs=20,
#                                   batch_size=512,
#                                   validation_data=(test_data, test_labels),
#                                   verbose=2)


l2_model=keras.models.Sequential()
l2_model.add(keras.layers.Dense(16,kernel_regularizer=keras.regularizers.l2(0.001),activation=tf.nn.relu,input_shape=(NUM_WORDS,)))
l2_model.add(keras.layers.Dense(16,kernel_regularizer=keras.regularizers.l2(0.001),activation=tf.nn.relu,input_shape=(NUM_WORDS,)))
l2_model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))
l2_model.compile(optimizer=tf.train.AdamOptimizer(),loss='binary_crossentropy',metrics=['accuracy','binary_crossentropy'])
l2_model_history=l2_model.fit(train_data,train_labels,epochs=20,batch_size=512,validation_data=(test_data,test_labels),verbose=2)



dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

dpt_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy','binary_crossentropy'])

dpt_model_history = dpt_model.fit(train_data, train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)



def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16,10))
    for name, history in histories:
        val=plt.plot(history.epoch, history.history['val_'+key],'--',label=name.title()+' Val')
        plt.plot(history.epoch,history.history[key],color=val[0].get_color(),label=name.title()+' Train')
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_'," ").title())
    plt.legend()
    plt.xlim([0, max(history.epoch)])
plot_history([('baseline',baseline_history),('smaller',smaller_history),('l2model',l2_model_history),('dpt_model',dpt_model_history)])
plt.show()



a = 1