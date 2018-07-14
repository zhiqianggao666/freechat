import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug


a = tf.constant(0)
b = 1
c=a+b

print(a,b,c)


constant = tf.constant([1, 2, 3])
tensor1 = constant * constant



# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100)) # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 构造一个线性模型
#
b = tf.Variable(tf.zeros([1]),name='bias')
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0),name='weight')
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

writer = tf.summary.FileWriter('./train_log')
writer.add_graph(tf.get_default_graph())

# 启动图 (graph)
sess = tf.Session()

with sess.as_default():
    sess.run(init)
    #sess.run(tf.global_variables_initializer())
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    #tensor_print = tf.Print(tensor,[tensor])
    #value = sess.run(tensor_print)
    value = sess.run(tensor1)
    #print(tensor.eval())
    #print(value)

    # 拟合平面
    for step in range(0, 201):
        sess.run(train)
        if step % 20 == 0:
            print (step, sess.run(W), sess.run(b), sess.run(loss))


        # 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]