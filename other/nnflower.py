from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import  os
from six.moves.urllib.request import urlopen

import  tensorflow as tf
import numpy as np
import tensorflow.contrib as contrib




IRIS_TRAINING="iris_training.csv"
IRIS_TRAINING_URL="http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST="iris_test.csv"
IRIS_TEST_URL="http://download.tensorflow.org/data/iris_test.csv"

tf.logging.set_verbosity(tf.logging.INFO)
if not os.path.exists(IRIS_TRAINING):
    raw = urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, 'wb') as f:
        f.write(raw)

if not os.path.exists(IRIS_TEST):
    raw=urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, 'wb') as f:
        f.write(raw)

training_set = contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TRAINING,target_dtype=np.int,features_dtype=np.float32)
test_set = contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TEST,target_dtype=np.int,features_dtype=np.float32)
feature_columns = [tf.feature_column.numeric_column("x",shape=[4])]
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[10,20,10],n_classes=3,model_dir="/tmp/iris_model")

train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":np.array(training_set.data)},y=np.array(training_set.target),num_epochs=None,shuffle=True)
classifier.train(input_fn=train_input_fn,steps=2000)

test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":np.array(test_set.data)},y=np.array(test_set.target),num_epochs=1,shuffle=False)
accuracy_score=classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy:{0:f}\n".format(accuracy_score))

new_samples = np.array([[6.4,3.2,4.5,1.5],[5.8,3.1,5.0,1.7]],dtype=np.float32)
predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":new_samples},num_epochs=1,shuffle=False)

predictions = list(classifier.predict(input_fn=predict_input_fn))
predicted_classes = [p["classes"] for p in predictions]
print("New Samples, Class Predictions: {}\n".format(predicted_classes))
