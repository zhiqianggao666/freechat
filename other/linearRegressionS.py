import tensorflow as tf
import numpy as np

def modelfn(features,labels,mode):
    w = tf.get_variable("w", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y_=w*features["x"]+b
    loss = tf.reduce_sum(tf.square(y_-labels))
    globalstep = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(globalstep,1))
    return tf.estimator.EstimatorSpec(mode=mode,predictions=y_,loss=loss,train_op=train)

x_train = np.array([1.,2.,3.,4.])
y_train = np.array([0.,-1,-2.,-3.])
x_eval = np.array([2.,5.,8.,1.])
y_eval = np.array([-1.01,-4.1,-7.,0.])

inputfn = tf.estimator.inputs.numpy_input_fn({"x":x_train},y_train,batch_size=4,num_epochs=None,shuffle=True)
trainfn = tf.estimator.inputs.numpy_input_fn({"x":x_train}, y_train, batch_size=4,num_epochs=1000,shuffle=False)
testfn = tf.estimator.inputs.numpy_input_fn({"x":x_eval}, y_eval, batch_size=4,num_epochs=1000, shuffle=False)

estimator = tf.estimator.Estimator(model_fn=modelfn)
estimator.train(input_fn=inputfn,steps=1000)

trainm = estimator.evaluate(input_fn=trainfn)
testm = estimator.evaluate(input_fn=testfn)
print (trainm,"=========================")
print (testm)