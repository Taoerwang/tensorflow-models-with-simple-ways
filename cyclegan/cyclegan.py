from __future__ import print_function, division
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
import os
import scipy


class CycleCOGAN:
    model_name = "cycle_gan"
    paper_name = ""
    paper_url = ""
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

    def discriminator_a(self, x):
        with tf.variable_scope('discriminator_a', reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(x, 512, tf.nn.leaky_relu, name='x_l')
            x = tf.layers.dense(x, 256, tf.nn.leaky_relu, name='x_2')
            x = tf.layers.dense(x, 128, tf.nn.leaky_relu, name='x_3')
            x = tf.layers.dense(x, 1, name='x_4')
        return x

    def discriminator_b(self, x):
        with tf.variable_scope('discriminator_b', reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(x, 512, tf.nn.leaky_relu, name='x_l')
            x = tf.layers.dense(x, 256, tf.nn.leaky_relu, name='x_2')
            x = tf.layers.dense(x, 128, tf.nn.leaky_relu, name='x_3')
            x = tf.layers.dense(x, 1, name='x_4')
        return x

    def generator_a2b(self, g):
        with tf.variable_scope('generator_a2b', reuse=tf.AUTO_REUSE):
            g = tf.layers.dense(g, 784, tf.nn.leaky_relu, name='g_1')
            g = tf.layers.batch_normalization(g, momentum=0.8)
            g = tf.layers.dense(g, 512, tf.nn.leaky_relu, name='g_2')
            g = tf.layers.batch_normalization(g, momentum=0.8)
            g = tf.layers.dense(g, 784, tf.nn.tanh, name='g_3')
        return g

    def generator_b2a(self, g):
        with tf.variable_scope('generator_b2a', reuse=tf.AUTO_REUSE):
            g = tf.layers.dense(g, 784, tf.nn.leaky_relu, name='g_1')
            g = tf.layers.batch_normalization(g, momentum=0.8)
            g = tf.layers.dense(g, 512, tf.nn.leaky_relu, name='g_2')
            g = tf.layers.batch_normalization(g, momentum=0.8)
            g = tf.layers.dense(g, 784, tf.nn.tanh, name='g_3')
        return g

    def build_model(self, learning_rate=0.0002):

        x_a = tf.placeholder(tf.float32, [None, self.img_rows * self.img_cols])
        x_b = tf.placeholder(tf.float32, [None, self.img_rows * self.img_cols])

        a2b = self.generator_a2b(x_a)
        b2a = self.generator_b2a(x_b)

        a2b2a = self.generator_b2a(a2b)
        b2a2b = self.generator_b2a(b2a)

        a_logit = self.discriminator_a(x_a)
        b_logit = self.discriminator_b(x_b)

        b2a_logit = self.discriminator_a(b2a)
        a2b_logit = self.discriminator_b(a2b)

        a2b2a_logit = self.discriminator_a(a2b2a)
        b2a2b_logit = self.discriminator_b(b2a2b)

        # cycle losses
        cyc_loss_a = tf.reduce_mean(tf.squared_difference(x_a, a2b2a))
        cyc_loss_b = tf.reduce_mean(tf.squared_difference(x_b, b2a2b))
        cyc_loss = cyc_loss_a + cyc_loss_b

        # g_a2b_loss
        g_loss_a_b = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(a2b_logit), logits=a2b_logit))
        g_loss_a2b = g_loss_a_b

        # g_b2a_loss
        g_loss_b_a = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(b2a_logit), logits=b2a_logit))
        g_loss_b2a = g_loss_b_a

        g_loss = cyc_loss * 100

        # d_loss_a
        d_loss_a_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(a_logit), logits=a_logit))
        d_loss_b2a = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(b2a_logit), logits=b2a_logit))

        d_loss_a = d_loss_a_real + d_loss_b2a

        # d_loss_b
        d_loss_b_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(b_logit), logits=b_logit))
        d_loss_a2b = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(a2b_logit), logits=a2b_logit))
        d_loss_b = d_loss_b_real + d_loss_a2b

        t_vars = tf.trainable_variables()
        d_vars_a = [var for var in t_vars if 'discriminator_a' in var.name]
        d_vars_b = [var for var in t_vars if 'discriminator_b' in var.name]
        g_vars_a = [var for var in t_vars if 'generator_a2b' in var.name]
        g_vars_b = [var for var in t_vars if 'generator_b2a' in var.name]
        g_vars = g_vars_a + g_vars_b

        g_optimizer = tf.train.AdamOptimizer(0.0002).minimize(g_loss, var_list=g_vars)
        g_optimizer_a2b = tf.train.AdamOptimizer(0.0002).minimize(g_loss_a2b, var_list=g_vars)
        g_optimizer_b2a = tf.train.AdamOptimizer(0.0002).minimize(g_loss_b2a, var_list=g_vars)

        d_optimizer_a = tf.train.AdamOptimizer(0.0002).minimize(d_loss_a, var_list=d_vars_a)
        d_optimizer_b = tf.train.AdamOptimizer(0.0002).minimize(d_loss_b, var_list=d_vars_b)

        return x_a, x_b,\
               g_loss, g_loss_a2b, g_loss_b2a, d_loss_a, d_loss_b, \
               g_optimizer, g_optimizer_a2b, g_optimizer_b2a, d_optimizer_a, d_optimizer_b,\
               x_a, x_b, a2b, b2a, a2b2a, b2a2b

    def train(self, train_steps=100000, batch_size=100, learning_rate=0.0002, save_model_numbers=3):
        x_a, x_b, \
        g_loss, g_loss_a2b, g_loss_b2a, d_loss_a, d_loss_b, \
        g_optimizer, g_optimizer_a2b, g_optimizer_b2a, d_optimizer_a, d_optimizer_b, \
        x_a, x_b, a2b, b2a, a2b2a, b2a2b = self.build_model(learning_rate)

        saver = tf.train.Saver(max_to_keep=save_model_numbers)
        if not os.path.exists('out/'):
            os.makedirs('out/')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for it in range(train_steps):
                index_1 = np.random.randint(0, self.img_counts, batch_size)
                index_2 = np.random.randint(0, self.img_counts, batch_size)

                batch_real_1 = self.train_images[index_1]
                batch_real_2 = self.train_images[index_2]

                batch_real_2 = np.reshape(batch_real_2, [-1, 28, 28])
                batch_real_2 = scipy.ndimage.interpolation.rotate(batch_real_2, 90, axes=(1, 2))
                batch_real_2 = np.reshape(batch_real_2, [-1, 784])

                # train D_a
                _, D_loss_a = sess.run([d_optimizer_a, d_loss_a], feed_dict={x_a: batch_real_1, x_b: batch_real_2})
                _, G_loss_b2a = sess.run([g_optimizer_b2a, g_loss_b2a],
                                         feed_dict={x_a: batch_real_1, x_b: batch_real_2})
                # train D_b
                _, D_loss_b = sess.run([d_optimizer_b, d_loss_b], feed_dict={x_a: batch_real_1, x_b: batch_real_2})
                _, G_loss_a2b = sess.run([g_optimizer_a2b, g_loss_a2b],
                                         feed_dict={x_a: batch_real_1, x_b: batch_real_2})

                _, G_loss = sess.run([g_optimizer, g_loss], feed_dict={x_a: batch_real_1, x_b: batch_real_2})

                if it % 1000 == 0:
                    print(it)
                    print("G_loss: " + str(G_loss))
                    print("D_loss_a: " + str(D_loss_a))
                    print("G_loss_a: " + str(G_loss_a2b))
                    print("D_loss_b: " + str(D_loss_b))
                    print("G_loss_b: " + str(G_loss_b2a))
                    print()

                    index_1 = np.random.randint(0, self.img_counts, batch_size)
                    index_2 = np.random.randint(0, self.img_counts, batch_size)

                    batch_real_1 = self.train_images[index_1]
                    batch_real_2 = self.train_images[index_2]

                    batch_real_2 = np.reshape(batch_real_2, [-1, 28, 28])
                    batch_real_2 = scipy.ndimage.interpolation.rotate(batch_real_2, 90, axes=(1, 2))
                    batch_real_2 = np.reshape(batch_real_2, [-1, 784])

                    test_a2b_image = sess.run(a2b, feed_dict={x_a: batch_real_1})
                    test_b2a_image = sess.run(b2a, feed_dict={x_b: batch_real_2})

                    build_a2b2a_image = sess.run(a2b2a, feed_dict={x_a: batch_real_1})
                    build_b2a2b_image = sess.run(b2a2b, feed_dict={x_b: batch_real_2})

                    r, c = 5, 5
                    fig, axs = plt.subplots(r, c)
                    cnt = 0
                    for i in range(r):
                        for j in range(c):
                            axs[i, j].imshow(np.reshape(batch_real_1[cnt], (28, 28)), cmap='gray')
                            axs[i, j].axis('off')
                            cnt += 1
                    fig.savefig("out/%d_real_a.png" % it)
                    plt.close()

                    r, c = 5, 5
                    fig, axs = plt.subplots(r, c)
                    cnt = 0
                    for i in range(r):
                        for j in range(c):
                            axs[i, j].imshow(np.reshape(batch_real_2[cnt], (28, 28)), cmap='gray')
                            axs[i, j].axis('off')
                            cnt += 1
                    fig.savefig("out/%d_real_b.png" % it)
                    plt.close()

                    r, c = 5, 5
                    fig, axs = plt.subplots(r, c)
                    cnt = 0
                    for i in range(r):
                        for j in range(c):
                            axs[i, j].imshow(np.reshape(test_a2b_image[cnt], (28, 28)), cmap='gray')
                            axs[i, j].axis('off')
                            cnt += 1
                    fig.savefig("out/%d_fake_a2b.png" % it)
                    plt.close()

                    r, c = 5, 5
                    fig, axs = plt.subplots(r, c)
                    cnt = 0
                    for i in range(r):
                        for j in range(c):
                            axs[i, j].imshow(np.reshape(test_b2a_image[cnt], (28, 28)), cmap='gray')
                            axs[i, j].axis('off')
                            cnt += 1
                    fig.savefig("out/%d_fake_b2a.png" % it)
                    plt.close()

                    r, c = 5, 5
                    fig, axs = plt.subplots(r, c)
                    cnt = 0
                    for i in range(r):
                        for j in range(c):
                            axs[i, j].imshow(np.reshape(build_a2b2a_image[cnt], (28, 28)), cmap='gray')
                            axs[i, j].axis('off')
                            cnt += 1
                    fig.savefig("out/%d_build_a2b2a.png" % it)
                    plt.close()

                    r, c = 5, 5
                    fig, axs = plt.subplots(r, c)
                    cnt = 0
                    for i in range(r):
                        for j in range(c):
                            axs[i, j].imshow(np.reshape(build_b2a2b_image[cnt], (28, 28)), cmap='gray')
                            axs[i, j].axis('off')
                            cnt += 1
                    fig.savefig("out/%d_build_b2a2b.png" % it)
                    plt.close()

                    print("over")
                # summary_str = sess.run(merged_summary_op,  feed_dict={x: batch_images, y: batch_labels})
                # summary_writer.add_summary(summary_str, i)
                # D:\py_project\untitled\tensorflow\gan_zoo_tensorflow\mlp > tensorboard - -logdir =./ log

    def restore_model(self):
        x_a, x_b, \
        g_loss, g_loss_a2b, g_loss_b2a, d_loss_a, d_loss_b, \
        g_optimizer, g_optimizer_a2b, g_optimizer_b2a, d_optimizer_a, d_optimizer_b, \
        x_a, x_b, a2b, b2a, a2b2a, b2a2b = self.build_model(0.0001)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model_file = tf.train.latest_checkpoint('ckpt/')
            saver.restore(sess, model_file)
            # write you code here


if __name__ == '__main__':
    data_sets = ['fashion_mnist', 'mnist']
    model = CycleCOGAN(data_sets[1])
    model.train()
    # model.restore_model()


