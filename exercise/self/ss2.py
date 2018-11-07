import tensorflow as tf

with tf.Session() as sess:
    const1=tf.constant([1,2,3,4,5,6,7,8])
    const1=tf.reshape(const1,shape=[-1,2])
    place1=tf.placeholder(dtype=tf.int32,shape=[None,2])
    print( sess.run(place1,feed_dict={place1:sess.run(const1)}))