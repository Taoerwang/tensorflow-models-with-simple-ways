from __future__ import print_function, division
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
import os

import prior_factory as prior


class AAE:
    model_name = "AAE"
    paper_name = "Adversarial Autoencoders(对抗自编码)"
    paper_url = "https://arxiv.org/abs/1511.05644"
    paper_chinese = "https://kingsleyhsu.github.io/2017/10/10/AAE/"
    git_hub = "https://github.com/hwalsuklee/tensorflow-mnist-AAE"
    data_sets = "MNIST and Fashion-MNIST"

    def __init__(self, data_name):
        self.data_name = data_name
        self.img_counts = 60000
        self.img_rows = 28
        self.img_cols = 28
        self.dim = 1
        self.noise_dim = 100

        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.load_data()
        self.train_images = np.reshape(self.train_images, (-1, self.img_rows * self.img_cols)) / 255
        self.test_images = np.reshape(self.test_images, (-1, self.img_rows * self.img_cols)) / 255

        self.train_labels_one_dim = self.train_labels
        self.test_labels_one_dim = self.test_labels
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

    def encoder(self, e, reuse=True):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            e = tf.layers.dense(e, 1024, tf.nn.relu, name='e_1')
            e = tf.layers.dense(e, 512, tf.nn.relu, name='e_2')
            e = tf.layers.dense(e, 256, tf.nn.relu, name='e_3')
            out = tf.layers.dense(e, 2, name='x_3')
        return out

    def decoder(self, d, reuse=True):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            d = tf.layers.dense(d, 128, tf.nn.relu, name='d_1')
            d = tf.layers.dense(d, 256, tf.nn.relu, name='d_2')
            d = tf.layers.dense(d, 512, tf.nn.relu, name='d_3')
            out = tf.layers.dense(d, 784, tf.nn.sigmoid, name='d_4')
        return out

    def discriminator(self, x, reuse=True):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(x, 128, tf.nn.relu, name='x_1')
            x = tf.layers.dense(x, 256, tf.nn.relu, name='x_2')
            out = tf.layers.dense(x, 1, name='x_3')
            return tf.sigmoid(out), out

    def build_model(self, learning_rate=0.001):

        x = tf.placeholder(tf.float32, [None, self.img_rows * self.img_cols])
        x_labels = tf.placeholder(tf.float32, shape=[None, 10])

        z_sample = tf.placeholder(tf.float32, shape=[None, 2])
        z_labels = tf.placeholder(tf.float32, shape=[None, 10])
        # g = tf.placeholder(tf.float32, [None, self.noise_dim])

        z = self.encoder(x)
        x_fake = self.decoder(z)
        e_d_loss = tf.reduce_mean(tf.squared_difference(x, x_fake))
        z_real = tf.concat([z_sample, z_labels], 1)
        z_fake = tf.concat([z, x_labels], 1)
        d_real, d_real_logits = self.discriminator(z_real)
        d_fake, d_fake_logits = self.discriminator(z_fake)
        # discriminator loss
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits, labels=tf.ones_like(d_real_logits)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits)))
        d_loss = d_loss_real + d_loss_fake
        # generator loss
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.ones_like(d_fake_logits)))

        d_loss = tf.reduce_mean(d_loss)
        g_loss = tf.reduce_mean(g_loss)

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        e_vars = [var for var in t_vars if 'encoder' in var.name]
        e_d_vars = [var for var in t_vars if "encoder" or "decoder" in var.name]

        # 优化器
        d_optimizer = tf.train.AdamOptimizer(learning_rate / 5).minimize(d_loss, var_list=d_vars)
        e_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=e_vars)
        e_d_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(e_d_loss, var_list=e_d_vars)

        return x, x_labels, z_sample, z_labels, \
            d_loss, g_loss, e_d_loss, \
            e_optimizer, d_optimizer, e_d_optimizer, \
            z, x_fake

    def train(self, train_steps=100000, batch_size=100, learning_rate=0.001, save_model_numbers=3):
        x, x_labels, z_sample, z_labels, \
            d_loss, g_loss, e_d_loss, \
            e_optimizer, d_optimizer, e_d_optimizer, \
            z, x_fake = self.build_model(learning_rate)
        saver = tf.train.Saver(max_to_keep=save_model_numbers)
        if not os.path.exists('out/'):
            os.makedirs('out/')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(train_steps):
                batch_index = np.random.randint(0, self.img_counts, batch_size)
                batch_real = self.train_images[batch_index]
                batch_labels = self.train_labels[batch_index]

                z_id_ = np.random.randint(0, 10, size=[batch_size])
                samples = prior.gaussian_mixture(batch_size, 2, label_indices=z_id_)
                z_id_one_hot_vector = np.zeros((batch_size, 10))
                z_id_one_hot_vector[np.arange(batch_size), z_id_] = 1

                sess.run(e_d_optimizer, feed_dict={x: batch_real, x_labels: batch_labels,
                                                   z_sample: samples, z_labels: z_id_one_hot_vector})
                sess.run(d_optimizer, feed_dict={x: batch_real, x_labels: batch_labels,
                                                 z_sample: samples, z_labels: z_id_one_hot_vector})
                sess.run(e_optimizer, feed_dict={x: batch_real, x_labels: batch_labels,
                                                 z_sample: samples, z_labels: z_id_one_hot_vector})

                if i % 1000 == 0:
                    e_d_loss_curr = sess.run(e_d_loss,
                                             feed_dict={x: batch_real, x_labels: batch_labels,
                                                        z_sample: samples, z_labels: z_id_one_hot_vector})
                    g_loss_curr = sess.run(g_loss,
                                           feed_dict={x: batch_real, x_labels: batch_labels,
                                                      z_sample: samples, z_labels: z_id_one_hot_vector})
                    d_loss_curr = sess.run(d_loss,
                                           feed_dict={x: batch_real, x_labels: batch_labels,
                                                      z_sample: samples, z_labels: z_id_one_hot_vector})

                    print('step: ' + str(i))
                    print('e_d_loss: ' + str(e_d_loss_curr))
                    print('G_loss:  ' + str(g_loss_curr))
                    print('D_loss:  ' + str(d_loss_curr))
                    print()
                    saver.save(sess, 'ckpt/mnist.ckpt', global_step=i)

                    r, c = 10, 10
                    samples = sess.run(x_fake, feed_dict={x: batch_real})
                    fig, axs = plt.subplots(r, c)
                    cnt = 0
                    for p in range(r):
                        for q in range(c):
                            axs[p, q].imshow(np.reshape(samples[cnt], (28, 28)), cmap='gray')
                            axs[p, q].axis('off')
                            cnt += 1
                    fig.savefig("out/%d.png" % i)
                    plt.close()

                    test_z = sess.run(z, feed_dict={x: self.train_images})
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    ax.scatter(test_z[:, 0], test_z[:, 1], c=self.train_labels_one_dim, s=10)
                    fig.savefig("out/%d_prediction.png" % i)
                    plt.close()

    def restore_model(self):
        x, x_labels, z_sample, z_labels, \
        d_loss, g_loss, e_d_loss, \
        e_optimizer, d_optimizer, e_d_optimizer, \
        z, x_fake = self.build_model(0.001)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model_file = tf.train.latest_checkpoint('ckpt/')
            saver.restore(sess, model_file)

            test_z = sess.run(z, feed_dict={x: self.train_images})
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(test_z[:, 0], test_z[:, 1], c=self.train_labels_one_dim, s=10)
            plt.show()


if __name__ == '__main__':
    datas = ['fashion_mnist', 'mnist']
    model = AAE(datas[1])
    model.train()
    # model.restore_model()
