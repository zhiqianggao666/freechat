import  tensorflow as tf
tf.enable_eager_execution()

tfe = tf.contrib.eager
from math import pi

def f(x,y):
    return tf.multiply(x*x,y)
grad_f=tfe.gradients_function(f)
c=grad_f(4.,4.)
print(c[0])
print(c[1])


import matplotlib.pyplot as plt
def f(x):
    return tf.square(tf.square(x))
def grad(f):
    return lambda x:tfe.gradients_function(f)(x)[0]
x=tf.lin_space(-100.,100.,200)
plt.plot(x,f(x),label='f')
plt.plot(x,grad(f)(x),label='first derivative')
plt.plot(x,grad(grad(f))(x),label='second derivative')
plt.plot(x,grad(grad(grad(f)))(x),label='third derivative')
plt.plot(x,grad(grad(grad(grad(f))))(x),label='fourth derivative')

plt.legend()
#plt.show()

def f(x,y):
    output=1
    for i in range(int(y)):
        output=tf.multiply(output,x)
    return output

def g(x,y):
    return tfe.gradients_function(f)(x,y)[0]

assert f(3.,2).numpy()==9.0
assert g(3.,2).numpy()==6.0
assert f(4.,3).numpy()==64.0
assert g(4.,3).numpy()==48.0

x=tf.ones((2,2))
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y=tf.reduce_sum(x)
    z=tf.multiply(y,y)

dz_dy=t.gradient(z,y)
assert dz_dy.numpy()==8.0

dz_dx=t.gradient(z,x)
for i in [0,1]:
    for j in [0,1]:
        assert dz_dx[i][j].numpy()==8.0
print(tf.reduce_sum(x))

x=tf.constant(1.0)
with tf.GradientTape() as t:
    with tf.GradientTape() as t2:
        t2.watch(x)
        y=x*x*x
    dy_dx=t2.gradient(y,x)

d2y_dx2=t.gradient(dy_dx,x)
assert dy_dx.numpy()==3.0
assert d2y_dx2.numpy()==6.0






a=1