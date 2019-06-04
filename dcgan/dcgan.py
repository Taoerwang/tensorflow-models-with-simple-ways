from __future__ import print_function, division
import tensorflow as tf
import tensorlayer as tl
from tensorflow import keras
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
import os


class DCGAN:
    model_name = "DCGAN"
    paper_name = ""
    paper_url = ""
    data_sets = "MNIST and Fashion-MNIST"

    def __init__(self, data_name):
        self.data_name = data_name
        self.img_counts = 60000
        self.img_rows = 28
        self.img_cols = 28
        self.img_dim = 1
        self.dim = 1
        self.noise_dim = 100

        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.load_data()
        # print(self.train_images.shape)
        self.train_images = np.reshape(self.train_images, (-1, self.img_rows, self.img_cols, self.img_dim)) / 255
        self.test_images = np.reshape(self.test_images, (-1, self.img_rows, self.img_cols, self.img_dim)) / 255
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

    def discriminator(self, inputs):
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):

            conv1 = tf.layers.conv2d(inputs=inputs,
                                     filters=32,
                                     kernel_size=[5, 5],
                                     padding="same",
                                     activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            conv2 = tf.layers.conv2d(inputs=pool1,
                                     filters=64,
                                     kernel_size=[5, 5],
                                     padding="same",
                                     activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            x = tf.reshape(pool2, [-1, 7 * 7 * 64])
            x = tf.layers.dense(x, 128, tf.nn.relu, name='x_l')
            x = tf.layers.dense(x, 128, tf.nn.relu, name='x_2')
            out = tf.layers.dense(x, 1, name='x_3')
        return out

    def generator(self, inputs, is_train=True):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            outputs = tf.layers.dense(inputs, 128 * 7 * 7, tf.nn.relu)
            outputs = tf.reshape(outputs, [-1, 7, 7, 128])
            outputs = tf.layers.batch_normalization(outputs, training=is_train)
            outputs = tf.layers.conv2d_transpose(outputs, 64, [3, 3], strides=(2, 2), padding='SAME')
            outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=is_train), name='outputs_2')
            outputs = tf.layers.conv2d_transpose(outputs, 1, [3, 3], strides=(2, 2), padding='SAME')
            outputs = tf.tanh(outputs, name='outputs')
        return outputs

    def build_model(self, learning_rate=0.0002):

        x = tf.placeholder(tf.float32, [None, self.img_rows, self.img_cols, self.img_dim])
        g = tf.placeholder(tf.float32, [None, self.noise_dim])

        g_sample = self.generator(g)
        d_logit_real = self.discriminator(x)
        d_logit_fake = self.discriminator(g_sample)

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_logit_real), logits=d_logit_real))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_logit_fake), logits=d_logit_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_logit_fake), logits=d_logit_fake))
        # 优化器
        d_optimizer = tf.train.AdamOptimizer(learning_rate, 0.5).minimize(d_loss, var_list=d_vars)
        g_optimizer = tf.train.AdamOptimizer(learning_rate, 0.5).minimize(g_loss, var_list=g_vars)
        return x, g, d_loss, g_loss, d_optimizer, g_optimizer, g_sample

    def train(self, train_steps=100000, batch_size=100, learning_rate=0.001, save_model_numbers=3):
        x, g, d_loss, g_loss, d_optimizer, g_optimizer, g_sample = self.build_model(learning_rate)
        saver = tf.train.Saver(max_to_keep=save_model_numbers)
        if not os.path.exists('out/'):
            os.makedirs('out/')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(train_steps):
                batch_index = np.random.randint(0, self.img_counts, batch_size)
                batch_real = self.train_images[batch_index]

                batch_fake = np.random.uniform(-1., 1., size=[batch_size, self.noise_dim])

                sess.run(d_optimizer, feed_dict={x: batch_real, g: batch_fake})
                sess.run(g_optimizer, feed_dict={x: batch_real, g: batch_fake})

                if i % 1000 == 0:
                    d_loss_curr = sess.run(d_loss, feed_dict={x: batch_real, g: batch_fake})
                    g_loss_curr = sess.run(g_loss, feed_dict={g: batch_fake})
                    print('step: ' + str(i))
                    print('D_loss: ' + str(d_loss_curr))
                    print('G_loss: ' + str(g_loss_curr))
                    print()
                    saver.save(sess, 'ckpt/mnist.ckpt', global_step=i)

                    r, c = 5, 5
                    batch_fake_img = np.random.uniform(-1., 1., size=[r * c, self.noise_dim])
                    samples = sess.run(g_sample, feed_dict={g: batch_fake_img})
                    fig, axs = plt.subplots(r, c)
                    cnt = 0
                    for p in range(r):
                        for q in range(c):
                            axs[p, q].imshow(np.reshape(samples[cnt], (28, 28)), cmap='gray')
                            axs[p, q].axis('off')
                            cnt += 1
                    fig.savefig("out/%d.png" % i)
                    plt.close()

    def restore_model(self):
        x, g, d_loss, g_loss, d_optimizer, g_optimizer, g_sample = self.build_model()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model_file = tf.train.latest_checkpoint('ckpt/')
            saver.restore(sess, model_file)

            # write you code here
            # r, c = 5, 5
            # batch_fake_img = np.random.uniform(-1., 1., size=[r * c, self.noise_dim])
            # samples = sess.run(g_sample, feed_dict={g: batch_fake_img})
            # fig, axs = plt.subplots(r, c)
            # cnt = 0
            # for p in range(r):
            #     for q in range(c):
            #         axs[p, q].imshow(np.reshape(samples[cnt], (28, 28)), cmap='gray')
            #         axs[p, q].axis('off')
            #         cnt += 1
            # plt.show()


if __name__ == '__main__':
    datas = ['fashion_mnist', 'mnist']
    model = DCGAN(datas[1])
    model.train()
    # model.restore_model()
