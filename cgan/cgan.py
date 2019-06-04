from __future__ import print_function, division
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
import os


class CGAN:
    model_name = "CGAN"
    paper_name = "Conditional generative adversarial nets"
    paper_url = "https://arxiv.org/pdf/1411.1784.pdf"
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

    def discriminator(self, x, y):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            x = tf.concat([x, y], 1)
            x = tf.layers.dense(x, 512, tf.nn.leaky_relu, name='x_l1')
            x = tf.layers.dense(x, 256, tf.nn.leaky_relu, name='x_2')
            out = tf.layers.dense(x, 1, name='x_3')
            return out

    def generator(self, g, y):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            g = tf.concat([g, y], 1)
            g = tf.layers.batch_normalization(tf.layers.dense(g, 256, tf.nn.leaky_relu, name='g_l'),
                                              momentum=0.8)
            g = tf.layers.batch_normalization(tf.layers.dense(g, 512, tf.nn.leaky_relu, name='g_2'),
                                              momentum=0.8)
            g = tf.layers.batch_normalization(tf.layers.dense(g, 1024, tf.nn.leaky_relu, name='g_3'),
                                              momentum=0.8)
            out = tf.layers.dense(g, 784, tf.nn.tanh, name='g_4')
            return out

    def build_model(self, learning_rate=0.0002):

        x = tf.placeholder(tf.float32, [None, self.img_rows * self.img_cols])
        g = tf.placeholder(tf.float32, [None, self.noise_dim])
        y_label = tf.placeholder(tf.float32, [None, 10])

        g_sample = self.generator(g, y_label)
        d_logit_real = self.discriminator(x, y_label)
        d_logit_fake = self.discriminator(g_sample, y_label)

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
        return x, g, y_label, \
            d_loss, g_loss, \
            d_optimizer, g_optimizer, \
            g_sample

    def train(self, train_steps=100000, batch_size=100, learning_rate=0.0002, save_model_numbers=3):
        x, g, y_label,\
            d_loss, g_loss, \
            d_optimizer, g_optimizer, \
            g_sample = self.build_model(learning_rate)
        saver = tf.train.Saver(max_to_keep=save_model_numbers)
        if not os.path.exists('out/'):
            os.makedirs('out/')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(train_steps):
                batch_index = np.random.randint(0, self.img_counts, batch_size)
                batch_real = self.train_images[batch_index]
                batch_label = self.train_labels[batch_index]

                batch_fake = np.random.uniform(-1., 1., size=[batch_size, self.noise_dim])

                sess.run(d_optimizer, feed_dict={x: batch_real, g: batch_fake, y_label: batch_label})
                sess.run(g_optimizer, feed_dict={x: batch_real, g: batch_fake, y_label: batch_label})

                if i % 1000 == 0:
                    d_loss_curr = sess.run(d_loss, feed_dict={x: batch_real, g: batch_fake, y_label: batch_label})
                    g_loss_curr = sess.run(g_loss, feed_dict={g: batch_fake, y_label: batch_label})
                    print('step: ' + str(i))
                    print('D_loss: ' + str(d_loss_curr))
                    print('G_loss: ' + str(g_loss_curr))
                    print()
                    saver.save(sess, 'ckpt/mnist.ckpt', global_step=i)

                    noise_labels = np.zeros([100, 10])
                    index = 0
                    for n in noise_labels:
                        n[index % 10] = 1
                        index += 1
                    noise_labels = np.array(noise_labels)

                    r, c = 10, 10
                    batch_fake_img = np.random.uniform(-1., 1., size=[r * c, self.noise_dim])
                    samples = sess.run(g_sample, feed_dict={g: batch_fake_img, y_label: noise_labels})
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
        x, g, y_label, \
        d_loss, g_loss, \
        d_optimizer, g_optimizer, \
        g_sample = self.build_model(0.001)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model_file = tf.train.latest_checkpoint('ckpt/')
            saver.restore(sess, model_file)
            # write your code
            #
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
    data_sets = ['fashion_mnist', 'mnist']
    model = CGAN(data_sets[1])
    model.train()
    # model.restore_model()
