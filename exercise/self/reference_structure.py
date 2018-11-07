import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline
import os
import shutil
import threading
import time
import win_unicode_console
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.contrib.tensorboard.plugins import projector
win_unicode_console.enable()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
minist = tf.keras.datasets.mnist
"""
坑s
1. 当使用Keras Model函数式模型的时候，可以引入其他层如tf.layers，也可以引入dataset API
2. 尽量不要使用Keras Sequential模式，因为完全产生一个keras模型，不利于引入其他layer，不利用自己掌控数据流，但是可以引入dataset API
3. model和layer是两个概念，model是keras的高层抽象，当使用session的时候，可以将keras的layer和tf.layers混合使用
4. 高层API有两种：1是keras自带的model.evaluate,2是keras转为estimator，然后使用estimator的evaluate
5. 常见错误维度不一致问题，原因：1. dataset没有进行batch，2. 输入feature和label没有进行reshape处理，一般都使用数组
6. 注意steps与epoch是不同的，一般steps_per_epoch=samples/batch，batch默认为32 
7. 内存与输入数据，输出数据，参数有关。参数只与模型本身有关。输入数据与batch大小有关，越大，内存越大，精度越高，每次迭代时间越长
8. timeline查看时间瓶颈
9. 调参步骤：learning rate->hidden units,batch size->hidden layers,rate decay
10. tensorboard --logdir=logs --port=8888,logs目录注意clear，不然有异常问题
11. python线程采用GIL机制，采用多进程并行调参
12. 采用FLAGs机制来组织工程
13. model与weights的save与restore，实现大数据分段迭代训练，第二次训练基于第一次save的权重，accuracy竟然比一次性训练高，有待研究
14. tensorboard embedding需要将x先reshape为一维数据后使用，否则无法显示，而且visual_embedding需要初始化为testset，否则是随机混乱的
15. tensorboard graph将from_tensor_slices数据加载为constant，因此不适合大数据，否则graph绘制极慢，应使用place holder替换
16. 推荐使用dataset中的feedable iterator，复杂但强大
17. validate的时候optimizer.minimize不能参与run，否则相当于每次都将test_set当成了训练集迭代了
18. reshape不确定的维度使用-1，但只能一个-1. place_holder不知道的维度使用None，运行时会动态分配内存
19. place_holder的feed数据只支持部分数据如python的numpy和list等，如果想支持tensor，需要先sess.run取出tensor再feed
todo:深入调参，estimator
"""
# Parameters
# ==================================================
# Model Hyperparameters
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate of optimizer (default: 0.001)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.2)")
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs (default: 5)")

# Training parameters
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("print_logs_every", 100, "Print logs after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", False, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("allow_timeline", False, "Allow timeline summary")
tf.flags.DEFINE_boolean("allow_tensorboard", True, "Allow tensorboard summary")
tf.flags.DEFINE_boolean("allow_tensorboard_embedding", True, "Allow tensorboard embedding")
tf.flags.DEFINE_boolean("allow_tensorboard_histogram", False, "Allow tensorboard histogram")
tf.flags.DEFINE_boolean("allow_matplot", False, "Allow matplot summary")
tf.flags.DEFINE_integer("debug_index", 1, "Current debug index id")
FLAGS = tf.flags.FLAGS
curr_path = os.getcwd()


def do_train(debug_index, data_slice_start,data_slice_end):
    print('=========preparing data directory')
    timeline_dir = os.path.join(curr_path, 'logs','timeline', str(debug_index))
    models_dir = os.path.join(curr_path, 'logs','models', str(debug_index))
    tensorboard_dir = os.path.join(curr_path, 'logs', 'tensorboard', str(debug_index))
    if FLAGS.allow_timeline:
        if not os.path.exists(timeline_dir):
            os.makedirs(timeline_dir)
    if FLAGS.allow_tensorboard or FLAGS.allow_tensorboard_embedding or FLAGS.allow_tensorboard_histogram:
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)


    print('=========loading data')
    (x_train, y_train), (x_test, y_test) = minist.load_data(path='mnist.zip')
    x_train, x_test = x_train / 255.0, x_test / 255.0

    if data_slice_end == 0:
        data_slice_start = 0
        data_slice_end = len(y_train)

    x_train = x_train.reshape((len(y_train), 784))[data_slice_start:data_slice_end]
    y_train = y_train.reshape((len(y_train), 1))[data_slice_start:data_slice_end]
    x_train, y_train=shuffle(x_train, y_train,random_state=0)
    x_test = x_test.reshape((len(x_test), 784))
    y_test = y_test.reshape((len(y_test), 1))

    print('=========creating session')
    config = None
    run_metadata = None
    options = None
    if FLAGS.allow_timeline:
        run_metadata = tf.RunMetadata()
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        config = tf.ConfigProto(graph_options=tf.GraphOptions(
            optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
    with tf.Session(config=config) as sess:
        print('=========converting dataset')
        x_train_ds=tf.placeholder(dtype=tf.float32,shape=np.shape(x_train))
        y_train_ds=tf.placeholder(dtype=tf.int32,shape=np.shape(y_train))

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train_ds, y_train_ds))
        train_dataset = train_dataset.map(lambda x, y: (tf.cast(x, dtype=tf.float32), tf.cast(y, dtype=tf.int32)))
        train_dataset = train_dataset.batch(FLAGS.batch_size).prefetch(2).repeat()

        x_test_ds=tf.placeholder(dtype=tf.float32,shape=np.shape(x_test))
        y_test_ds=tf.placeholder(dtype=tf.int32,shape=np.shape(y_test))
        val_dataset = tf.data.Dataset.from_tensor_slices((x_test_ds, y_test_ds))
        val_dataset = val_dataset.map(lambda x, y: (tf.cast(x, dtype=tf.float32), tf.cast(y, dtype=tf.int32)))
        val_dataset = val_dataset.batch(len(x_test)).prefetch(2).repeat()

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        features, labels = iterator.get_next()
        training_iterator = train_dataset.make_initializable_iterator()
        val_iterator = val_dataset.make_initializable_iterator()


        if FLAGS.allow_tensorboard:
            image_dis = x_train[:3]
            image_dis = np.reshape(image_dis, (3, 28, 28, 1))
            tf.summary.image('image', tf.constant(value=image_dis, dtype=tf.float32))

        if FLAGS.allow_tensorboard_embedding:
            visual_embedding=tf.Variable(initial_value=x_test[:1000],trainable=False,name='visual_embedding')

        def print_variable_count():
            total = 0
            for variable in tf.trainable_variables():
                variable_p = 1
                for dim in variable.shape:
                    variable_p *= dim
                total += variable_p
            print('total trainable variables: {}, size: {:.4f}M'.format(total, int(total) * 4 / (1024 * 1024)))

        def model_fn(input, is_training):
            print('=========creating models')
            def layer1(x):
                return tf.layers.dense(x, 512, activation=tf.nn.relu)

            weights = {
                'ww2': tf.get_variable(name='ww2', shape=[512, 10], initializer=tf.glorot_uniform_initializer())
            }
            bias = {
                'bb2': tf.get_variable(name='bb2', shape=[10], initializer=tf.zeros_initializer())
            }

            def layer2(x):
                z2 = tf.add(tf.matmul(x, weights['ww2']), bias['bb2'])
                a2 = tf.nn.softmax(z2)
                return a2

            x = layer1(input)
            if (is_training):
                x = tf.layers.dropout(x, FLAGS.dropout_keep_prob)
            predictions=layer2(x)
            print_variable_count()
            return predictions

        def visualize_tensorboard_embedding(sess, training_steps):
            print('=========visualizing tensorboard embeddings.')
            meta_file_path=os.path.join(tensorboard_dir ,'embeddings','metadata.tsv')
            if not os.path.exists(os.path.join(tensorboard_dir ,'embeddings')):
                os.makedirs(os.path.join(tensorboard_dir ,'embeddings'))

            def create_sprite_image(images):
                if isinstance(images, list):
                    images = np.array(images)
                img_h = images.shape[1]
                img_w = images.shape[2]
                n_plots = int(np.ceil(np.sqrt(images.shape[0])))

                spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

                for i in range(n_plots):
                    for j in range(n_plots):
                        this_filter = i * n_plots + j
                        if this_filter < images.shape[0]:
                            this_img = images[this_filter]
                            spriteimage[i * img_h:(i + 1) * img_h,
                            j * img_w:(j + 1) * img_w] = this_img

                return spriteimage

            def vector_to_matrix_mnist(mnist_digits):
                return np.reshape(mnist_digits, (-1, 28, 28))

            def invert_grayscale(mnist_digits):
                return 1 - mnist_digits

            to_visualise = x_test[:1000]
            to_visualise = vector_to_matrix_mnist(to_visualise)
            to_visualise = invert_grayscale(to_visualise)

            sprite_image = create_sprite_image(to_visualise)

            plt.imsave(os.path.join(tensorboard_dir,'embeddings','mnist_sprite.png'), sprite_image, cmap='gray')
            #plt.imshow(sprite_image, cmap='gray')

            with open(meta_file_path, 'w') as f:
                test_len=len(y_test[:1000])
                for i in range(test_len):
                    f.write(str(y_test[i]) + '\n')

            summary_writer=tf.summary.FileWriter(os.path.join(tensorboard_dir,'embeddings'),graph=sess.graph)
            config=projector.ProjectorConfig()
            embedding=config.embeddings.add()
            embedding.tensor_name=visual_embedding.name
            embedding.metadata_path=meta_file_path
            embedding.sprite.image_path=os.path.join(tensorboard_dir,'embeddings','mnist_sprite.png')
            embedding.sprite.single_image_dim.extend([28,28])
            projector.visualize_embeddings(summary_writer=summary_writer,config=config)
            saver=tf.train.Saver([visual_embedding])
            saver.save(sess,os.path.join(tensorboard_dir,'embeddings','model.ckpt'),training_steps)
            summary_writer.close()


        predictions = model_fn(features, True)
        accuracy = tf.metrics.accuracy(labels, tf.argmax(predictions, axis=1, output_type=tf.int32))[1]
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels, predictions))

        if FLAGS.allow_tensorboard:
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)
        if FLAGS.allow_tensorboard_histogram:
            collection_keys = sess.graph.get_all_collection_keys()
            trainable_variables = tf.get_default_graph().get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            tf.summary.histogram('weights1', [trainable_variables[0]])
            tf.summary.histogram('weights2', trainable_variables[2])
            tf.summary.histogram('bias1', [trainable_variables[1]])
            tf.summary.histogram('bias2', trainable_variables[3])

        train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss,
                                                                                      global_step=tf.train.get_or_create_global_step())

        print('=========loading model from last ckpt')
        if os.path.exists(models_dir):
            try:
                ckpt=tf.train.get_checkpoint_state(models_dir)
            except ValueError:
                print('models does not exist')
            finally:
                pass
            if ckpt:
                print('=========importing graph from last ckpt')
                saver = tf.train.Saver()
                #saver=tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
                saver.restore(sess,ckpt.model_checkpoint_path)
            else:
                print('=========no pre-trained models')

        print('=========initing global/local variables')
        if not ckpt:
            sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        training_handle = sess.run(training_iterator.string_handle())
        val_handle = sess.run(val_iterator.string_handle())

        merged = tf.constant(1, dtype=tf.int32, name='tensorboard_constant')
        if FLAGS.allow_tensorboard or FLAGS.allow_tensorboard_histogram:
            merged = tf.summary.merge_all()
            train_summary_writer = tf.summary.FileWriter(os.path.join(tensorboard_dir, 'train'), graph=sess.graph)
            val_summary_writer = tf.summary.FileWriter(os.path.join(tensorboard_dir, 'val'), graph=sess.graph)

        train_accuracy_list = []
        train_loss_list = []
        temp_loss_list = []

        val_accuracy_list = []
        val_loss_list = []
        saver = tf.train.Saver()
        steps = int(len(y_train) / FLAGS.batch_size)
        sess.run(training_iterator.initializer,feed_dict={x_train_ds:x_train,y_train_ds:y_train})
        sess.run(val_iterator.initializer,feed_dict={x_test_ds:x_test,y_test_ds:y_test})
        print('=========running epochs')
        for i in range(FLAGS.num_epochs):
            for j in range(steps):

                train_merged_value, train_accuracy_value, train_loss_value, _ = sess.run(
                    [merged, accuracy, loss, train_op], feed_dict={handle: training_handle}, run_metadata=run_metadata,
                    options=options)
                temp_loss_list.append(train_loss_value)

                if FLAGS.allow_tensorboard or FLAGS.allow_tensorboard_histogram:
                    train_summary_writer.add_summary(train_merged_value, i * steps + j)

                if (i * steps + j) % FLAGS.print_logs_every == 0:
                    print('epoch:{}/{}     ,steps:{}/{},     accuracy:{:.4f}%,     loss:{:.4f}'.format(i + 1,
                                                                                                       FLAGS.num_epochs,
                                                                                                       j, steps,
                                                                                                       train_accuracy_value * 100,
                                                                                                       train_loss_value))

                if (i * steps + j) % FLAGS.evaluate_every == 0:
                    train_loss_list.append(np.mean(temp_loss_list))
                    temp_loss_list = []
                    train_accuracy_list.append(train_accuracy_value)
                    val_merged_value, val_accuracy_value, val_loss_value = sess.run(
                        [merged, accuracy, loss],
                        feed_dict={handle: val_handle})
                    print('+++++++++++val: epoch:{}/{}     ,steps:{}/{},     accuracy:{:.4f}%,     loss:{:.4f}'.format(
                        i + 1, FLAGS.num_epochs, j, steps, val_accuracy_value * 100, val_loss_value))
                    val_loss_list.append(val_loss_value)
                    val_accuracy_list.append(val_accuracy_value)
                    if FLAGS.allow_tensorboard or FLAGS.allow_tensorboard_histogram:
                        val_summary_writer.add_summary(val_merged_value, i * steps + j)

                if (i * steps + j) % FLAGS.checkpoint_every == 0:
                    saver.save(sess, os.path.join(models_dir, 'model.ckpt'),
                               global_step=tf.train.get_or_create_global_step())

        if FLAGS.allow_timeline:
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()

        if FLAGS.allow_tensorboard or FLAGS.allow_tensorboard_histogram:
            train_summary_writer.close()
            val_summary_writer.close()

        if FLAGS.allow_tensorboard_embedding:
            visualize_tensorboard_embedding(sess, FLAGS.num_epochs * steps)

    if FLAGS.allow_timeline:
        with open(os.path.join(timeline_dir, 'timeline.json'), 'w') as wd:
            wd.write(ctf)

    if FLAGS.allow_matplot:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        fig.suptitle('Learning Curves')
        axes[0].set_ylabel('Loss', fontsize=14)
        axes[0].plot(train_loss_list, label='Train Loss')
        axes[0].plot(val_loss_list, 'r--', label='Val Loss')

        axes[1].set_ylabel('Accuracy', fontsize=14)
        axes[1].set_xlabel('Epoch*Steps', fontsize=14)
        axes[1].plot(train_accuracy_list, label='Train Accuracy')
        axes[1].plot(val_accuracy_list, 'r--', label='Val Accuracy')
        plt.show()


def main(argvs):
    print('>>>>>>start training for index:{}'.format(FLAGS.debug_index))
    if FLAGS.allow_tensorboard or FLAGS.allow_tensorboard_embedding or FLAGS.allow_tensorboard_histogram:
        if os.path.exists(os.path.join(curr_path, 'logs','tensorboard', str(FLAGS.debug_index))):
            shutil.rmtree(os.path.join(curr_path, 'logs', 'tensorboard',str(FLAGS.debug_index)))

    if FLAGS.allow_timeline:
        if os.path.exists(os.path.join(curr_path, 'logs', 'timeline', str(FLAGS.debug_index))):
            shutil.rmtree(os.path.join(curr_path, 'logs', 'timeline',str(FLAGS.debug_index)))
    data_slice_start=50000
    data_slice_end=0
    do_train(FLAGS.debug_index,data_slice_start,data_slice_end)
    print('<<<<<<end training for index:{}'.format(FLAGS.debug_index))


if __name__ == '__main__':
    tf.app.run()

'''
tensorboard --logdir=tensorboard --port=8888
python -m keras-tflayer-Dataset-Session --debug_index=1 --learning_rate=0.01 --num_epochs=5
python -m keras-tflayer-Dataset-Session --debug_index=2 --learning_rate=0.001 --num_epochs=5
'''