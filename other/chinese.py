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



def map_preprocess_func1(x):
    ret= "*".join(x.decode('utf-8'))
    #print(ret)
    return ret

def map_funct1(line):
    return tf.py_func(map_preprocess_func1, [line], tf.string)



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


dataset2 = tf.data.Dataset.from_tensor_slices(list(u'我爱南京'))
dataset2 = dataset2.batch(1)
iter2 = dataset2.make_one_shot_iterator()


with tf.Session() as sess:
    print(sess.run(str).decode())
    print(sess.run(iter.get_next())[0][0][0].decode())
    print(sess.run(iter2.get_next()))
    