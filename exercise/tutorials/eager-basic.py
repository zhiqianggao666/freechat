import  tensorflow as tf
tf.enable_eager_execution()

print(tf.add(1,2))
print(tf.add([1,2],[3,4]))
print(tf.square(5))
print(tf.reduce_sum([1,2,3]))
print(tf.encode_base64('hello world'))
print(tf.square(2)+tf.square(3))
x=tf.matmul([[1]],[[2,3]])
print(x)
print(x.shape)
print(x.dtype)
import numpy as np
ndarray = np.ones([3, 3])
print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)

print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())

x=tf.random_uniform([3,3])
print('is there a gpu available')
print(tf.test.is_gpu_available(0))
print('is the tensor on gpu#0: ')
print(x.device.endswith('GPU:0'))

def time_matmul(x):
    return tf.matmul(x,x)
print('On CPU')
with tf.device('CPU:0'):
    x=tf.random_uniform([1000,1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)

if tf.test.is_gpu_available():
    with tf.device("GPU:0"):
        x=tf.random_uniform([1000,1000])
        assert x.device.endswith('GPU:0')
        time_matmul(x)

ds_tensors=tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6])
import tempfile
_,filename=tempfile.mkstemp()
with open(filename,'w') as f:
    f.write("""Line 1
    Line 2
    Line 3
    """)
print('filename:',filename)
ds_file = tf.data.TextLineDataset(filename)

ds_tensors=ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)

print('Elements of ds_tensors:')
for x in ds_tensors:
    print(x)

print('\nElements in ds_file:')
for x in ds_file:
    print(x)
