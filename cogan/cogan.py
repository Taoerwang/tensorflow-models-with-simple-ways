from __future__ import print_function, division
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
import os
import scipy


class COGAN:
    model_name = "COGAN"
    paper_name = "Coupled Generative Adversarial Network"
    paper_url = "https://arxiv.org/abs/1606.07536"
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

    def sample_Z(self, m, n):
        return np.random.uniform(-1., 1., size=[m, n])

    def discriminator(self, x):
        with tf.variable_scope('dis_', reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(x, 128, tf.nn.leaky_relu, name='x_1')
            x = tf.layers.dense(x, 50, tf.nn.leaky_relu, name='x_2')
            out = tf.layers.dense(x, 1, name='x_3')
            return out

    def discriminator_1(self, x):
        with tf.variable_scope('discriminator_1', reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(x, 512, tf.nn.leaky_relu, name='x_l')
            x = tf.layers.dense(x, 256, tf.nn.leaky_relu, name='x_2')
            return x

    def discriminator_2(self, x):
        with tf.variable_scope('discriminator_2', reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(x, 512, tf.nn.leaky_relu, name='x_l')
            x = tf.layers.dense(x, 256, tf.nn.leaky_relu, name='x_2')
            return x

    def generator(self, g):
        with tf.variable_scope('gen_', reuse=tf.AUTO_REUSE):
            g = tf.layers.dense(g, 256, tf.nn.leaky_relu, name='g_2')
            g = tf.layers.batch_normalization(g, momentum=0.8)
            g = tf.layers.dense(g, 512, tf.nn.leaky_relu, name='g_3')
            g = tf.layers.batch_normalization(g, momentum=0.8)
        return g

    def generator_1(self, g):
        with tf.variable_scope('generator_1', reuse=tf.AUTO_REUSE):
            g = tf.layers.dense(g, 512, tf.nn.leaky_relu, name='g_2')
            g = tf.layers.batch_normalization(g, momentum=0.8)
            out = tf.layers.dense(g, 784, tf.nn.tanh, name='g_3')
        return out

    def generator_2(self, g):
        with tf.variable_scope('generator_2', reuse=tf.AUTO_REUSE):
            g = tf.layers.dense(g, 512, tf.nn.leaky_relu, name='g_2')
            g = tf.layers.batch_normalization(g, momentum=0.8)
            out = tf.layers.dense(g, 784, tf.nn.tanh, name='g_3')
        return out

    def build_model(self, learning_rate=0.0002):

        x_1 = tf.placeholder(tf.float32, [None, self.img_rows * self.img_cols])
        x_2 = tf.placeholder(tf.float32, [None, self.img_rows * self.img_cols])
        z = tf.placeholder(tf.float32, [None, 100])
        # g = tf.placeholder(tf.float32, [None, self.noise_dim])

        # z_g = self.generator(z)

        x_1_g = self.generator_1(self.generator(x_1))
        d_1_fake = self.discriminator(self.discriminator_1(x_1_g))
        d_1_real = self.discriminator(self.discriminator_1(x_1))

        x_2_g = self.generator_2(self.generator(x_2))
        d_2_fake = self.discriminator(self.discriminator_2(x_2_g))
        d_2_real = self.discriminator(self.discriminator_2(x_2))

        d_loss_real_1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_1_real), logits=d_1_real))
        d_loss_fake_1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_1_fake), logits=d_1_fake))
        d_loss_1 = d_loss_real_1 + d_loss_fake_1
        g_loss_1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_1_fake), logits=d_1_fake))

        d_loss_real_2 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_2_real), logits=d_2_real))
        d_loss_fake_2 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_2_fake), logits=d_2_fake))
        d_loss_2 = d_loss_real_2 + d_loss_fake_2
        g_loss_2 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_2_fake), logits=d_2_fake))

        d_loss = d_loss_1 + d_loss_2
        g_loss = g_loss_1 + g_loss_2

        t_vars = tf.trainable_variables()
        d_vars_1 = [var for var in t_vars if 'discriminator_1' in var.name]\
                        + [var for var in t_vars if 'dis_' in var.name]
        g_vars_1 = [var for var in t_vars if 'generator_1' in var.name]\
                        + [var for var in t_vars if 'gen_' in var.name]
        d_vars_2 = [var for var in t_vars if 'discriminator_2' in var.name]\
                        + [var for var in t_vars if 'dis_' in var.name]
        g_vars_2 = [var for var in t_vars if 'generator_2' in var.name]\
                   + [var for var in t_vars if 'gen_' in var.name]

        d_vars = [var for var in t_vars if 'discriminator_1' in var.name]\
                        + [var for var in t_vars if 'discriminator_2' in var.name]\
                        + [var for var in t_vars if 'dis_' in var.name]
        g_vars = [var for var in t_vars if 'generator_1' in var.name] \
                 + [var for var in t_vars if 'generator_2' in var.name]\
                   + [var for var in t_vars if 'gen_' in var.name]

        d_optimizer_1 = tf.train.AdamOptimizer(0.0002, 0.5).minimize(d_loss_1, var_list=d_vars_1)
        g_optimizer_1 = tf.train.AdamOptimizer(0.0002, 0.5).minimize(g_loss_1, var_list=g_vars_1)
        d_optimizer_2 = tf.train.AdamOptimizer(0.0002, 0.5).minimize(d_loss_2, var_list=d_vars_2)
        g_optimizer_2 = tf.train.AdamOptimizer(0.0002, 0.5).minimize(g_loss_2, var_list=g_vars_2)

        d_optimizer = tf.train.AdamOptimizer(0.0002, 0.5).minimize(d_loss, var_list=d_vars)
        g_optimizer = tf.train.AdamOptimizer(0.0002, 0.5).minimize(g_loss, var_list=g_vars)
        # 优化器

        x_1_2 = self.generator_2(self.generator(x_1))
        x_2_1 = self.generator_1(self.generator(x_2))

        return x_1, x_2, z, \
               d_loss_1, g_loss_1, d_loss_2, g_loss_2, \
               d_optimizer_1, g_optimizer_1, d_optimizer_2, g_optimizer_2, \
               x_1_g, x_2_g, \
               d_loss, g_loss, d_optimizer, g_optimizer, x_1_2, x_2_1

    def train(self, train_steps=100000, batch_size=100, learning_rate=0.0002, save_model_numbers=3):
        x_1, x_2, z, \
        d_loss_1, g_loss_1, d_loss_2, g_loss_2, \
        d_optimizer_1, g_optimizer_1, d_optimizer_2, g_optimizer_2, \
        x_1_g, x_2_g, \
        d_loss, g_loss, d_optimizer, g_optimizer, x_1_2, x_2_1 = self.build_model(learning_rate)

        saver = tf.train.Saver(max_to_keep=save_model_numbers)
        if not os.path.exists('out/'):
            os.makedirs('out/')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # merged_summary_op = tf.summary.merge_all()
            # summary_writer = tf.summary.FileWriter('log/mnist_with_summaries', sess.graph)
            for i in range(train_steps):
                index_1 = np.random.randint(0, self.img_counts, batch_size)
                index_2 = np.random.randint(0, self.img_counts, batch_size)

                batch_real_1 = self.train_images[index_1]
                batch_real_2 = self.train_images[index_2]

                batch_real_2 = np.reshape(batch_real_2, [-1, 28, 28])
                batch_real_2 = scipy.ndimage.interpolation.rotate(batch_real_2, 90, axes=(1, 2))
                batch_real_2 = np.reshape(batch_real_2, [-1, 784])

                z_noise = self.sample_Z(batch_size, 100)

                # sess.run(d_optimizer_1,
                #          feed_dict={x_1: batch_real_1, x_2: batch_real_2, z: z_noise})
                # sess.run(g_optimizer_1,
                #          feed_dict={x_1: batch_real_1, x_2: batch_real_2, z: z_noise})
                #
                # sess.run(d_optimizer_2,
                #          feed_dict={x_1: batch_real_1, x_2: batch_real_2, z: z_noise})
                # sess.run(g_optimizer_2,
                #          feed_dict={x_1: batch_real_1, x_2: batch_real_2, z: z_noise})

                sess.run(d_optimizer,
                         feed_dict={x_1: batch_real_1, x_2: batch_real_2, z: z_noise})
                sess.run(g_optimizer,
                         feed_dict={x_1: batch_real_1, x_2: batch_real_2, z: z_noise})

                if i % 1000 == 0:
                    d_loss_curr = sess.run(d_loss,
                                           feed_dict={x_1: batch_real_1, x_2: batch_real_2, z: z_noise})
                    g_loss_curr = sess.run(g_loss,
                                           feed_dict={x_1: batch_real_1, x_2: batch_real_2, z: z_noise})

                    print('Iter: {}'.format(i))
                    print('D_loss: {:.4}'.format(d_loss_curr))
                    print('G_loss: {:.4}'.format(g_loss_curr))
                    print()

                    saver.save(sess, 'ckpt/mnist.ckpt', global_step=i)

                    test_noise = np.random.uniform(-1., 1., size=[batch_size, 100])

                    r, c = 10, 10
                    samples = batch_real_1
                    fig, axs = plt.subplots(r, c)
                    cnt = 0
                    for p in range(r):
                        for q in range(c):
                            axs[p, q].imshow(np.reshape(samples[cnt], (28, 28)), cmap='gray')
                            axs[p, q].axis('off')
                            cnt += 1
                    fig.savefig("out/%d_1_real.png" % i)
                    plt.close()

                    r, c = 10, 10
                    samples = batch_real_2
                    fig, axs = plt.subplots(r, c)
                    cnt = 0
                    for p in range(r):
                        for q in range(c):
                            axs[p, q].imshow(np.reshape(samples[cnt], (28, 28)), cmap='gray')
                            axs[p, q].axis('off')
                            cnt += 1
                    fig.savefig("out/%d_2_real.png" % i)
                    plt.close()

                    r, c = 10, 10
                    samples = sess.run(x_1_g, feed_dict={x_1: batch_real_1})
                    fig, axs = plt.subplots(r, c)
                    cnt = 0
                    for p in range(r):
                        for q in range(c):
                            axs[p, q].imshow(np.reshape(samples[cnt], (28, 28)), cmap='gray')
                            axs[p, q].axis('off')
                            cnt += 1
                    fig.savefig("out/%d_1_fake.png" % i)
                    plt.close()

                    r, c = 10, 10
                    samples = sess.run(x_2_g, feed_dict={x_2: batch_real_2})
                    fig, axs = plt.subplots(r, c)
                    cnt = 0
                    for p in range(r):
                        for q in range(c):
                            axs[p, q].imshow(np.reshape(samples[cnt], (28, 28)), cmap='gray')
                            axs[p, q].axis('off')
                            cnt += 1
                    fig.savefig("out/%d_2_fake.png" % i)
                    plt.close()
                    print('over')


    def restore_model(self):
        x_1, x_2, z, \
        d_loss_1, g_loss_1, d_loss_2, g_loss_2, \
        d_optimizer_1, g_optimizer_1, d_optimizer_2, g_optimizer_2, \
        x_1_g, x_2_g, \
        d_loss, g_loss, d_optimizer, g_optimizer, x_1_2, x_2_1 = self.build_model()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model_file = tf.train.latest_checkpoint('ckpt/')
            saver.restore(sess, model_file)

            i = 0

            index_1 = np.random.randint(0, self.img_counts, 100)
            index_2 = np.random.randint(0, self.img_counts, 100)

            batch_real_1 = self.train_images[index_1]
            batch_real_2 = self.train_images[index_2]

            batch_real_2 = np.reshape(batch_real_2, [-1, 28, 28])
            batch_real_2 = scipy.ndimage.interpolation.rotate(batch_real_2, 90, axes=(1, 2))
            batch_real_2 = np.reshape(batch_real_2, [-1, 784])

            r, c = 10, 10
            samples = batch_real_1
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for p in range(r):
                for q in range(c):
                    axs[p, q].imshow(np.reshape(samples[cnt], (28, 28)), cmap='gray')
                    axs[p, q].axis('off')
                    cnt += 1
            fig.savefig("test/%d_1_real.png" % i)
            plt.close()

            r, c = 10, 10
            samples = batch_real_2
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for p in range(r):
                for q in range(c):
                    axs[p, q].imshow(np.reshape(samples[cnt], (28, 28)), cmap='gray')
                    axs[p, q].axis('off')
                    cnt += 1
            fig.savefig("test/%d_2_real.png" % i)
            plt.close()

            r, c = 10, 10
            samples = sess.run(x_1_g, feed_dict={x_1: batch_real_1})
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for p in range(r):
                for q in range(c):
                    axs[p, q].imshow(np.reshape(samples[cnt], (28, 28)), cmap='gray')
                    axs[p, q].axis('off')
                    cnt += 1
            fig.savefig("test/%d_1_fake.png" % i)
            plt.close()

            r, c = 10, 10
            samples = sess.run(x_2_g, feed_dict={x_2: batch_real_2})
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for p in range(r):
                for q in range(c):
                    axs[p, q].imshow(np.reshape(samples[cnt], (28, 28)), cmap='gray')
                    axs[p, q].axis('off')
                    cnt += 1
            fig.savefig("test/%d_2_fake.png" % i)
            plt.close()

            r, c = 10, 10
            samples = sess.run(x_1_2, feed_dict={x_1: batch_real_1})
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for p in range(r):
                for q in range(c):
                    axs[p, q].imshow(np.reshape(samples[cnt], (28, 28)), cmap='gray')
                    axs[p, q].axis('off')
                    cnt += 1
            fig.savefig("test/%d_1_2.png" % i)
            plt.close()

            r, c = 10, 10
            samples = sess.run(x_2_1, feed_dict={x_2: batch_real_2})
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for p in range(r):
                for q in range(c):
                    axs[p, q].imshow(np.reshape(samples[cnt], (28, 28)), cmap='gray')
                    axs[p, q].axis('off')
                    cnt += 1
            fig.savefig("test/%d_2_1.png" % i)
            plt.close()
            print('over')


if __name__ == '__main__':
    data_sets = ['fashion_mnist', 'mnist']
    model = COGAN(data_sets[1])
    # model.train()
    model.restore_model()
