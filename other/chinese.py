#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

def preprocess_func(x):
    ret= "*".join(x.decode('utf-8'))
    print(ret)
    return ret

str = tf.py_func(
        preprocess_func,
        [tf.constant(u"我爱，南京")],
        tf.string)



def map_preprocess_func(x):
    ret= "*".join(x.decode('utf-8'))
    print(ret)
    return ret

def map_funct(line):
    return tf.py_func(map_preprocess_func, [line], tf.string)

dataset = tf.data.TextLineDataset('/home/tizen/share/charmpy/tf/cmn-eng/b.txt')
dataset = dataset.map(map_funct)
dataset = dataset.batch(1)
inter = dataset.make_one_shot_iterator()


with tf.Session() as sess:
    sess.run(str)
    sess.run(inter.get_next())
    