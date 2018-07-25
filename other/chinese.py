#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

def preprocess_func(x):
    ret= "*".join(x.decode('utf-8'))
    #print(ret)
    return ret

str = tf.py_func(
        preprocess_func,
        [tf.constant(u"我爱，南京")],
        tf.string)

with tf.Session() as sess:
    print('11111111111111:',sess.run(str).decode())
########################################################################################

def map_preprocess_func1(x):
    ret= "*".join(x.decode('utf-8'))
    #print(ret)
    return ret

def map_funct1(line):
    return tf.py_func(map_preprocess_func1, [line], tf.string)

dataset = tf.data.TextLineDataset('/home/tizen/share/charmpy/tf/cmn-eng/b.txt')
dataset = dataset.map(map_funct1)
dataset = dataset.batch(1)
iter = dataset.make_one_shot_iterator()
with tf.Session() as sess:
    print('222222222222:',sess.run(iter.get_next()))
########################################################################################

def map_preprocess_func2(x):
    c=[word for word in x.decode('utf-8')]
    a=1
    return c,a

def map_funct2(line):
    return tf.py_func(map_preprocess_func2, [line], [tf.string, tf.int64])


dataset = tf.data.TextLineDataset('/home/tizen/share/charmpy/tf/cmn-eng/b.txt')
dataset = dataset.map(map_funct2)
dataset = dataset.batch(1)
iter = dataset.make_one_shot_iterator()
with tf.Session() as sess:
    print('33333333333:',sess.run(iter.get_next())[0][0][0].decode())
########################################################################################

dataset2 = tf.data.Dataset.from_tensor_slices(list(u'我爱南京'))
dataset2 = dataset2.batch(1)
iter2 = dataset2.make_one_shot_iterator()
with tf.Session() as sess:
    print('44444444444444444:',sess.run(iter2.get_next()))
    
########################################################################################
#from_tensor_slices, all data should be same size
#from_generator, size can be different


src_data = [list(u'我爱南京'), list(u'你好啊啊啊')]
tgt_data = [list(u'你是谁'), list(u'嗯嗯')]
dataset3 = tf.data.Dataset.zip((tf.data.Dataset.from_generator(lambda: src_data, tf.string),
                                       tf.data.Dataset.from_generator(lambda: tgt_data, tf.string)))
dataset3 = dataset3.batch(1)
iter3 = dataset3.make_one_shot_iterator()
with tf.Session() as sess:
    print('55555555555:',sess.run(iter3.get_next()))
    
########################################################################################
dataset4=tf.data.Dataset.from_tensor_slices(list(u'我还喜欢她,怎么办'))
#dataset4=tf.data.Dataset.from_tensor_slices([list(u'我还喜欢她,怎么办')])
#dataset4 = dataset4.batch(1)
iter4 = dataset4.make_one_shot_iterator()

with tf.Session() as sess:
    print('666666666666:',sess.run(iter4.get_next()))


########################################################################################
inputdata = tf.placeholder(dtype=tf.string, shape=[None])
dataset4=tf.data.Dataset.from_tensor_slices([inputdata])
#iter5 = dataset4.make_one_shot_iterator()
iter5 = dataset4.make_initializable_iterator()

with tf.Session() as sess:
    print('7777777777:',sess.run(iter5.initializer, feed_dict={inputdata : list('我爱中国')}))
    print('7777777777:',sess.run(iter5.get_next(), feed_dict={inputdata : list('我爱中国')}))