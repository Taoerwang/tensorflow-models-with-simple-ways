from __future__ import print_function, division
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.utils import np_utils
import tensorflow.contrib.slim as slim


class HighWayNet:
    model_name = "HighWayNet"
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

    def load_data(self):
        if self.data_name == "fashion_mnist":
            data_sets = keras.datasets.fashion_mnist
        elif self.data_name == "mnist":
            data_sets = keras.datasets.mnist
        else:
            data_sets = keras.datasets.mnist
        return data_sets.load_data()

    def highwayUnit(self, x, filters):

        h = tf.layers.conv2d(inputs=x,
                             filters=filters,
                             kernel_size=[5, 5],
                             padding="same")

        t = tf.layers.conv2d(inputs=x,
                             filters=filters,
                             kernel_size=[5, 5],
                             bias_initializer=tf.constant_initializer(-1.0),
                             activation=tf.nn.sigmoid,
                             padding="same")

        output = h*t + x*(1.0-t)
        return output

    def classifier(self, input, reuse=False):
        with tf.variable_scope('classifier', reuse):
            x = tf.reshape(input, (-1, 28, 28, 1))
            layer1 = tf.layers.conv2d(inputs=x,
                                      filters=32,
                                      kernel_size=[5, 5],
                                      padding="same")
            layer1 = tf.nn.relu(layer1)

            layer1 = self.highwayUnit(layer1, 32)
            layer1 = self.highwayUnit(layer1, 32)
            layer1 = self.highwayUnit(layer1, 32)
            layer1 = self.highwayUnit(layer1, 32)
            layer1 = self.highwayUnit(layer1, 32)
            layer1 = tf.layers.max_pooling2d(inputs=layer1, pool_size=[2, 2], strides=2)

            layer1 = self.highwayUnit(layer1, 32)
            layer1 = self.highwayUnit(layer1, 32)
            layer1 = self.highwayUnit(layer1, 32)
            layer1 = tf.layers.max_pooling2d(inputs=layer1, pool_size=[2, 2], strides=2)
            # layer1 = tf.layers.max_pooling2d(inputs=layer1, pool_size=[2, 2], strides=2)

            x = tf.reshape(layer1, [-1, 7 * 7 * 32])
            x = tf.layers.dense(x, 256, tf.nn.relu, name='x_2')
            out = tf.layers.dense(x, 10, name='x_3')

            return out

    def build_model(self, learning_rate):

        x = tf.placeholder(tf.float32, [None, self.img_rows * self.img_cols])
        y = tf.placeholder(tf.float32, [None, 10])

        y_prediction = self.classifier(x)

        t_vars = tf.trainable_variables()
        c_vars = [var for var in t_vars if 'classifier' in var.name]

        c_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_prediction))
        c_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss=c_loss, var_list=c_vars)

        correct_prediction = tf.equal(tf.arg_max(y_prediction, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # tf.summary.scalar("loss", c_loss)
        return x, y, c_loss, c_optimizer, accuracy

    def train(self, train_steps=11000, batch_size=100, learning_rate=0.0001, save_model_numbers=3):
        x, y, c_loss, c_optimizer, accuracy = self.build_model(learning_rate)
        saver = tf.train.Saver(max_to_keep=save_model_numbers)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # merged_summary_op = tf.summary.merge_all()
            # summary_writer = tf.summary.FileWriter('log/mnist_with_summaries', sess.graph)
            for i in range(train_steps):
                batch_index = np.random.randint(0, self.img_counts, batch_size)
                batch_images = self.train_images[batch_index]
                batch_labels = self.train_labels[batch_index]

                sess.run(c_optimizer, feed_dict={x: batch_images, y: batch_labels})
                if i % 1000 == 0:
                    train_loss, train_accuracy = sess.run([c_loss, accuracy],
                                                          feed_dict={x: batch_images, y: batch_labels})
                    print('step' + str(i) + ' train_accuracy:' + str(train_accuracy) + ' loss:' + str(train_loss))
                    saver.save(sess, 'ckpt/mnist.ckpt', global_step=i)

                # summary_str = sess.run(merged_summary_op, feed_dict={x: batch_images, y: batch_labels})
                # summary_writer.add_summary(summary_str, i)
                # D:\py_project\untitled\tensorflow\gan_zoo_tensorflow\mlp > tensorboard - -logdir =./ log
            # test_accuracy = accuracy.eval(feed_dict={x: self.test_images, y: self.test_labels})
            # print('test_accuracy' + str(test_accuracy))

    def restore_model(self):
        x, y, c_loss, c_optimizer, accuracy = self.build_model(0.0001)
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
                test_accuracy_ = sess.run(accuracy, feed_dict={x: test_img, y: test_labels})
                a += 1000
                b += 1000
                print('test_accuracy' + str(test_accuracy_))
                test_accuracy += test_accuracy_
            print(test_accuracy)
            print(test_accuracy / 10)


if __name__ == '__main__':
    model = HighWayNet("fashion_mnist")
    model.train()
    # model.restore_model()

