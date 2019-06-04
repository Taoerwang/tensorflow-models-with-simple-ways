from __future__ import print_function, division
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.utils import np_utils
import tensorflow.contrib.slim as slim


# 加上原始的数据

class MLP:
    model_name = "MLP"
    data_sets = "MNIST and Fashion-MNIST"

    def __init__(self, data_name):
        self.data_name = data_name
        self.img_counts = 60000
        self.img_rows = 28
        self.img_cols = 28
        self.dim = 1

        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.load_data()
        self.train_images = np.reshape(self.train_images, (-1, self.img_rows * self.img_cols)) / 255
        self.test_images = np.reshape(self.test_images, (-1, self.img_rows * self.img_cols)) / 255
        self.train_labels = np_utils.to_categorical(self.train_labels)
        self.test_labels = np_utils.to_categorical(self.test_labels)

        self.num_classes = 1000
        self.width_multiplier = 1

    def load_data(self):
        if self.data_name == "fashion_mnist":
            data_sets = keras.datasets.fashion_mnist
        elif self.data_name == "mnist":
            data_sets = keras.datasets.mnist
        else:
            data_sets = keras.datasets.mnist
        return data_sets.load_data()

    # def classifier(self, x, reuse=False):
    #     with tf.variable_scope('classifier', reuse):
    #         x = tf.layers.dense(x, 256, tf.nn.relu, name='x_l')
    #         # x1 = tf.concat([x1, x], 1)
    #         a = x
    #         for i in range(50):
    #
    #             x2 = tf.layers.dense(x, 256, tf.nn.relu)
    #             x = a + x2
    #
    #         out = tf.layers.dense(x, 10, name='x_7')
    #         return out

    def res_depthwise_separable_conv(self, inputs, expansion_ratio, output_dim, shortcut):
        """ Helper function to build the depth-wise separable convolution layer.
        """
        bottleneck_dim = round(expansion_ratio * inputs.get_shape().as_list()[-1])

        x = slim.convolution2d(inputs, bottleneck_dim, kernel_size=[1, 1], activation_fn=tf.nn.relu6)
        x = tf.layers.batch_normalization(x)

        # skip pointwise by setting num_outputs=None
        x = slim.separable_convolution2d(x, None, [3, 3], 1, activation_fn=tf.nn.relu6)
        x = tf.layers.batch_normalization(x)

        x = slim.convolution2d(x, output_dim, kernel_size=[1, 1], activation_fn=None)
        x = tf.layers.batch_normalization(x)

        if shortcut:
            in_dim = int(inputs.get_shape().as_list()[-1])
            if in_dim != output_dim:
                ins = tf.layers.conv2d(inputs=inputs,  filters=output_dim, kernel_size=[1, 1], padding="same")
                ins = tf.layers.batch_normalization(ins)
                x = ins + x
            else:
                x = inputs + x

        return x

    def classifier(self, x, keep_prob, reuse=False):
        with tf.variable_scope('classifier', reuse):
            x = tf.reshape(x, (-1, 28, 28, 1))

            x = tf.layers.conv2d(x, 64, [3, 3], padding="same", activation=tf.nn.relu)
            x = tf.layers.batch_normalization(x)
            x = self.res_depthwise_separable_conv(x, 5, 64, True)
            x = self.res_depthwise_separable_conv(x, 5, 64, True)
            x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)

            x = tf.reshape(x, [-1, 14 * 14 * 64])
            x = tf.layers.dense(x, 1000, tf.nn.relu)
            x = tf.nn.dropout(x, keep_prob)
            out = tf.layers.dense(x, 10)
            return out

    def build_model(self, learning_rate):

        x = tf.placeholder(tf.float32, [None, self.img_rows * self.img_cols])
        y = tf.placeholder(tf.float32, [None, 10])
        # z = tf.placeholder(tf.float32, [None, 10])
        keep_prob = tf.placeholder(tf.float32)

        y_prediction = self.classifier(x, keep_prob)

        t_vars = tf.trainable_variables()
        c_vars = [var for var in t_vars if 'classifier' in var.name]

        c_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_prediction))
        c_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss=c_loss, var_list=c_vars)

        correct_prediction = tf.equal(tf.arg_max(y_prediction, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("loss", c_loss)
        return x, y, keep_prob, c_loss, c_optimizer, accuracy

    def train(self, epochs=10, batch_size=100, learning_rate=0.0001, save_model_numbers=6):
        x, y, keep_prob, c_loss, c_optimizer, accuracy = self.build_model(learning_rate)
        saver = tf.train.Saver(max_to_keep=save_model_numbers)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # merged_summary_op = tf.summary.merge_all()
            # summary_writer = tf.summary.FileWriter('log/mnist_with_summaries', sess.graph)

            for epoch in range(epochs):
                a = 0
                b = 100
                for i in range(600):
                    batch_images = self.train_images[list(range(a, b))]
                    batch_labels = self.train_labels[list(range(a, b))]
                    a += 100
                    b += 100
                    sess.run(c_optimizer, feed_dict={x: batch_images, y: batch_labels, keep_prob: 0.5})
                    if i == 599:
                        train_loss, train_accuracy = sess.run([c_loss, accuracy],
                                                              feed_dict={x: batch_images,
                                                                         y: batch_labels,
                                                                         keep_prob: 0.5})
                        print(
                            'epoch' + str(epoch) + ' train_accuracy' + str(train_accuracy) + ' loss:' + str(train_loss))
                        saver.save(sess, 'ckpt/mnist.ckpt', global_step=epoch)

                # summary_str = sess.run(merged_summary_op,  feed_dict={x: batch_images, y: batch_labels})
                # summary_writer.add_summary(summary_str, i)
                # D:\py_project\untitled\tensorflow\gan_zoo_tensorflow\mlp > tensorboard - -logdir =./ log
            # test_accuracy = sess.run(accuracy, feed_dict={x: self.test_images, y: self.test_labels, keep_prob: 1.0})
            # print('test_accuracy' + str(test_accuracy))

    def restore_model(self):
        x, y, keep_prob, c_loss, c_optimizer, accuracy = self.build_model(0.0001)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model_file = tf.train.latest_checkpoint('ckpt/')
            saver.restore(sess, model_file)
            # batch_index = np.random.randint(0, 10000, 2000)
            a = 0
            b = 1000
            test_accuracy = 0
            for i in range(0, 10):
                test_img = self.test_images[list(range(a, b))]
                test_labels = self.test_labels[list(range(a, b))]
                test_accuracy_ = sess.run(accuracy, feed_dict={x: test_img, y: test_labels, keep_prob: 1.0})
                a += 1000
                b += 1000
                print('test_accuracy' + str(test_accuracy_))
                test_accuracy += test_accuracy_
            print(test_accuracy)
            print(test_accuracy / 10)

    def restart(self, epochs, lr, keep_probs):
        x, y, keep_prob, c_loss, c_optimizer, accuracy = self.build_model(lr)
        saver = tf.train.Saver(max_to_keep=5)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model_file = tf.train.latest_checkpoint('ckpt/')
            saver.restore(sess, model_file)

            for epoch in range(epochs):
                for i in range(600):
                    batch_index = np.random.randint(0, self.img_counts, 100)
                    batch_images = self.train_images[batch_index]
                    batch_labels = self.train_labels[batch_index]
                    sess.run(c_optimizer, feed_dict={x: batch_images, y: batch_labels, keep_prob: keep_probs})
                    if i == 599:
                        train_loss, train_accuracy = sess.run([c_loss, accuracy],
                                                              feed_dict={x: batch_images,
                                                                         y: batch_labels,
                                                                         keep_prob: keep_probs})
                        print(
                            'epoch' + str(epoch) + ' train_accuracy' + str(train_accuracy) + ' loss:' + str(train_loss))
                        saver.save(sess, 'ckpt/mnist.ckpt', global_step=epoch)


if __name__ == '__main__':
    model = MLP("fashion_mnist")
    # model.train()
    model.restore_model()
    # model.restart(10, 0.0001, 0.5)
