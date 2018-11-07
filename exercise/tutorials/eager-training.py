import  tensorflow as tf
tfe=tf.contrib.eager
tf.enable_eager_execution()

x=tf.zeros([10,10])
x+=2
print(x)

v=tfe.Variable(1.0)
assert v.numpy()==1.0
v.assign(3.0)
assert v.numpy()==3.0
v.assign(tf.square(v))
assert v.numpy()==9.0

class Model(object):
    def __init__(self):
        self.W=tfe.Variable(5.0)
        self.b=tfe.Variable(0.0)
    def __call__(self,x):
        return self.W*x+self.b
model = Model()
assert model(3.0).numpy()==15.0

def loss(predicted_y,desired_y):
    return tf.reduce_mean(tf.square(predicted_y-desired_y))
TRUE_W=3.0
TRUE_b=2.0
NUM_EXAMPLES=1000
inputs=tf.random_normal(shape=[NUM_EXAMPLES])
noise=tf.random_normal(shape=[NUM_EXAMPLES])
outputs=inputs*TRUE_W+TRUE_b+noise

import matplotlib.pyplot as plt
plt.scatter(inputs,outputs,c='b')
plt.scatter(inputs,model(inputs),c='r')
#plt.show()
print('Current Loss :')
print(loss(model(inputs),outputs).numpy())

def train(model,inputs,outputs,learing_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dW,db=t.gradient(current_loss,[model.W,model.b])
    model.W.assign_sub(learing_rate*dW)
    model.b.assign_sub(learing_rate*db)

model = Model()
Ws,bs=[],[]
epochs=range(10)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs), outputs)
    train(model,inputs,outputs,learing_rate=0.1)
    print('Epoch %2d: W=%1.2f, W=%1.2f, loss=2.5%f'%(epoch, Ws[-1], bs[-1], current_loss))

plt.clf()
plt.plot(epochs,Ws,'r')
plt.plot(bs,'b')
plt.plot(epochs,'g')
plt.plot([TRUE_W]*len(epochs),'r--')
plt.plot([TRUE_b]*len(epochs),'b--')
plt.legend(['W','b','true_w','true_b'])
plt.show()