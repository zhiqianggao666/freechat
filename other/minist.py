import tensorflow as tf
import tensorlayer as tl

sess = tf.InteractiveSession()

with tf.variable_scope("a_variable_scope",reuse=False) as scope:
    initializer = tf.constant_initializer(value=3)
    var3 = tf.get_variable(name='var3', shape=[1], dtype=tf.float32, initializer=initializer)
    var4 = tf.get_variable(name='var4', shape=[1], dtype=tf.float32, initializer=initializer)

    scope.reuse_variables()

    var3_reuse = tf.get_variable(name='var3', )
    var4_reuse = tf.get_variable(name='var4', )

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var3.name)  # a_variable_scope/var3:0
    print(var3_reuse.name)  # a_variable_scope/var3:0
    print(var4.name)  # a_variable_scope/var4:0
    print(var4_reuse.name)  # a_variable_scope/var4_1:0

X_train, y_tran, X_val, y_val,X_test,y_test=tl.files.load_mnist_dataset(shape=(-1,784))

x=tf.placeholder(tf.float32,shape=[None,784], name='x')
y_=tf.placeholder(tf.int64,shape=[None],name='y_')

network =  tl.layers.InputLayer(x,name='input')
network = tl.layers.DropoutLayer(network,keep=0.8,name='drop1')
network = tl.layers.DenseLayer(network,800,tf.nn.relu,name='relu1')
network = tl.layers.DropoutLayer(network,keep = 0.5,name='drop2')
network = tl.layers.DenseLayer(network,800,tf.nn.relu,name='relu2')
network = tl.layers.DropoutLayer(network,keep = 0.5,name='drop3')

network = tl.layers.DenseLayer(network,n_units=10,act=tf.identity,name='output')

y=network.outputs
cost = tl.cost.cross_entropy(y,y_,name='cost')
correct_prediction=tf.equal(tf.argmax(y,1),y_)
acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
y_op=tf.argmax(tf.nn.softmax(y),1)
e2=tf.equal(y_op,tf.argmax(y,1))
train_params = network.all_params
train_op=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_params)

tl.layers.initialize_global_variables(sess)

network.print_params()
network.print_layers()

#tl.utils.fit(sess,network,train_op,cost,X_train,y_tran,x,y_,acc=acc,batch_size=500,n_epoch=1,print_freq=5,X_val=X_val,y_val=y_val,eval_train=False)
# tl.utils.test(sess,network,acc,X_test,y_test,x,y_,batch_size=None,cost=cost)
#
# tl.files.save_npz(network.all_params,name='model.npz')
# sess.close()

batch_size = 30
n_step = int(len(y_tran)/batch_size)

import  time
###============= train
n_epoch = 50
for epoch in range(n_epoch):
	
    epoch_time = time.time()
    ## train an epoch
    total_err, n_iter = 0, 0
    for X, Y in tl.iterate.minibatches(inputs=X_train, targets=y_tran, batch_size=batch_size, shuffle=True):
        step_time = time.time()
        feed_dict = {x: X, y_: Y}
        feed_dict.update(network.all_drop)
        _, err = sess.run([train_op, cost],
                          feed_dict=feed_dict)

        if n_iter % 200 == 0:
            print("Epoch[%d/%d] step:[%d/%d] loss:%f took:%.5fs" % (epoch, n_epoch, n_iter, n_step, err, time.time() - step_time))

        a1,a2,a3,a4,a5,a6=sess.run([correct_prediction,acc,y,y_op,cost,e2],feed_dict=feed_dict)

        total_err += err
        n_iter += 1

 

    print("Epoch[%d/%d] averaged loss:%f took:%.5fs" % (epoch, n_epoch, total_err/n_iter, time.time()-epoch_time))

    tl.files.save_npz(network.all_params, name='n.npz', sess=sess)

sess.close()
