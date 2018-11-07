from __future__ import print_function,division,absolute_import
import tensorflow as tf
dataset1=tf.data.Dataset.from_tensor_slices(tf.random_uniform([4,10]))
print(dataset1.output_shapes)
print(dataset1.output_types)

dataset2=tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]),tf.random_uniform([4,10],maxval=100,dtype=tf.int32)))
print(dataset2.output_types)
print(dataset2.output_shapes)

dataset3=tf.data.Dataset.zip((dataset1,dataset2))
print(dataset3.output_shapes)
print(dataset3.output_types)

dataset=tf.data.Dataset.from_tensor_slices({'a':tf.random_uniform([4]),'b':tf.random_uniform([4,100],maxval=100,dtype=tf.int32)})
print(dataset.output_types)
print(dataset.output_shapes)



max_value=tf.placeholder(tf.int64,shape=[])
dataset=tf.data.Dataset.range(max_value)
iterator=dataset.make_initializable_iterator()
with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={max_value:10})
    for i in range(10):
        value=sess.run(iterator.get_next())


training_dataset=tf.data.Dataset.range(10).map(
    lambda x: x+tf.random_uniform([],-10,10,tf.int64)
)
validation_dataset=tf.data.Dataset.range(5)
iterator=tf.data.Iterator.from_structure(training_dataset.output_types,training_dataset.output_shapes)
next_element=iterator.get_next()
training_init_op=iterator.make_initializer(training_dataset)
validation_init_op=iterator.make_initializer(validation_dataset)
for _ in range(1):
    with tf.Session() as sess:
        sess.run(training_init_op)
        for _ in range(10):
            print(sess.run(next_element))
        sess.run(validation_init_op)
        for _ in range(5):
            print(sess.run(next_element))



training_dataset=tf.data.Dataset.range(100).map(
    lambda x:x+tf.random_uniform([],-10,10,tf.int64)
).repeat()
validation_dataset=tf.data.Dataset.range(50)
handle=tf.placeholder(tf.string,shape=[])
iterator=tf.data.Iterator.from_string_handle(handle,training_dataset.output_types,training_dataset.output_shapes)
next_element=iterator.get_next()
training_iterator=training_dataset.make_one_shot_iterator()
validation_iterator=validation_dataset.make_initializable_iterator()
with tf.Session() as sess:
    training_handle=sess.run(training_iterator.string_handle())
    validation_handle=sess.run(validation_iterator.string_handle())
    for _ in range(100):
        sess.run(next_element,feed_dict={handle:training_handle})
    sess.run(validation_iterator.initializer)
    for _ in range(50):
        sess.run(next_element,feed_dict={handle:validation_handle})

print('===========================================')
dataset=tf.data.Dataset.range(4).batch(2)
iterator=dataset.make_initializable_iterator()
next_element=iterator.get_next()
result=tf.add(next_element,next_element)
with tf.Session() as sess:
    sess.run(iterator.initializer)
    while True:
        try:
            print(sess.run(result))
        except tf.errors.OutOfRangeError:
            print('End of the dataset')
            break

dataset1=tf.data.Dataset.from_tensor_slices(tf.random_uniform([4,10]))
dataset2=tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]),tf.random_uniform([4,100])))
dataset3=tf.data.Dataset.zip((dataset1,dataset2))
iterator=dataset3.make_initializable_iterator()
with tf.Session() as sess:
    sess.run(iterator.initializer)
    #next1,(next2,next3)=iterator.get_next()
    print(sess.run(iterator.get_next()))


saveable=tf.contrib.data.make_saveable_from_iterator(iterator)
tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS,saveable)
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(iterator.initializer)
    saver.save(sess,'./iterator')
with tf.Session() as sess:
    saver.restore(sess,'./iterator')

import  numpy as np
with np.load('test.npy') as data:
    features=data['features']
    labels=data['labels']
    assert features.shape[0] == labels.shape[0]
    #wate memory
    dataset=tf.data.Dataset.from_tensor_slices((features,labels))

    feature_placeholder=tf.placeholder(features.dtype,features.shape)
    label_placeholder=tf.placeholder(labels.dtype,labels.shape)
    dataset=tf.data.Dataset.from_tensor_slices((feature_placeholder, label_placeholder))
    iterator=dataset.make_initializable_iterator()
    with tf.Session() as sess:
        sess.run(iterator.initializer,feed_dict={feature_placeholder:features,label_placeholder:labels})

filenames=['test1.tfrecord','test2.tfrecord']
dataset=tf.data.TFRecordDataset(filenames)

filenames=['test1.txt','test2.txt']
dataset=tf.data.Dataset.from_tensor_slices(filenames)
dataset=dataset.flat_map(lambda filename:(tf.data.TextLineDataset(filename).skip(1).filter(lambda  line:tf.not_equal(tf.substr(line,0,1),'#'))))

filenames=['test1.csv','test2.csv']
record_defaults=[tf.float32]*8
from tensorflow.contrib.data  import CsvDataset
dataset=CsvDataset(filenames,record_defaults,header=False,select_cols=[2,4])

def parse_function(example_proto):
    features={'image':tf.FixedLenFeature((),tf.string,default_value=''),
              'label':tf.FixedLenFeature((),tf.int64,default_value=0)
              }
    parsed_feature=tf.parse_single_example(example_proto,features)
    return parsed_feature['image'],parsed_feature['label']


filenames=['test1.tfrecord','test2.tfrecord']
dataset=tf.data.TFRecordDataset(filenames)
dataset=dataset.map(parse_function)


def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label
filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])
labels = tf.constant([0, 37, ...])
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)

filenames=['test1.jpg','test2.jpg']
labels=[2,1]
def _read_py_function(filename,label):
    image_decode=filename
    return image_decode,label

def _resize_function(image_decoded,label):
    image_decoded=None
    image_resized=tf.image.resize_images(image_decoded,[28,28])
    return image_resized,label

dataset=tf.data.Dataset.from_tensor_slices((filenames,labels))
dataset=dataset.map(lambda filename,label:tuple(tf.py_funct(_read_py_function,[filename,label],[tf.uint8,label.dtype])))
dataset=dataset.map(_resize_function)

inc_dataset=tf.data.Dataset.range(100)
dec_dataset=tf.data.Dataset.range(0,-100,-1)
dataset=tf.data.Dataset.zip(inc_dataset,dec_dataset)
batched_dataset=dataset.batch(4)
iterator=batched_dataset.make_one_shot_iterator()
next_element=iterator.get_next()
with tf.Session() as sess:
    print(sess.run(next_element))

dataset=tf.data.Dataset.range(100)
dataset=dataset.map(lambda x:tf.fill([tf.cast(x,tf.int32),x]))
dataset=dataset.padded_batch(4,padded_shapes=[None])
dataset=dataset.shuffle(buffer_size=100)

with tf.train.MonitoredTrainingSession() as sess:
    sess.run()

def input_func(dataset):
    num_epochs=10
    filenames=['test1.tfrecord','test2.tfrecord']
    dataset=tf.data.TFRecordDataset(filenames)
    def parser(record):
        keys_to_features={'image_data':tf.FixedLenFeature((),tf.string,default_value=''),
                          'date_time':tf.FixedLenFeature(().tf.int64,default_value=''),
                          'label':tf.FixedLenFeature((),tf.int64,default_value=tf.zeros([],dtype=tf.int64))}
        parsed=tf.parse_single_example(record,keys_to_features)
        image=tf.image.decode_jpeg(parsed['image_data'])
        image=tf.reshape(image,[299,299,1])
        label=tf.cast(parsed['label'],tf.int32)
        return {'image_data':image,'date_time':parsed['date_time'],'label':label}

    dataset=dataset.map(parser)
    dataset=dataset.shuffle(buffer_size=10000)
    dataset=dataset.batch(32)
    dataset=dataset.repeat(num_epochs)
    return dataset