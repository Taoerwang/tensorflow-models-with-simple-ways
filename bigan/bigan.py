from __future__ import print_function, division
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
import os


class BiGAN:
    model_name = "BiGAN"
    paper_name = "Adversarial Feature Learning(对抗特征学习)"
    paper_url = "https://arxiv.org/abs/1605.09782"
    chinese = "https://juejin.im/entry/5a36299851882538e2259b80"
    data_sets = "MNIST and Fashion-MNIST"

    def __init__(self, data_name):
        self.data_name = data_name
        self.img_counts = 60000
        self.img_rows = 28
        self.img_cols = 28
        self.dim = 1
        self.noise_dim = 100
        self.z_dim = 2

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

    def encoder(self, e):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            e = tf.layers.dense(e, 512, tf.nn.leaky_relu, name='e_1')
            e = tf.layers.dense(e, 512, tf.nn.leaky_relu, name='e_2')
            out = tf.layers.dense(e, self.z_dim, name='e_3')
        return out

    def decoder(self, d):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            d = tf.layers.dense(d, 256, tf.nn.leaky_relu, name='d_2')
            d = tf.layers.dense(d, 512, tf.nn.leaky_relu, name='d_3')
            out = tf.layers.dense(d, 784, tf.nn.tanh, name='d_4')
        return out

    def discriminator(self, z, x):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            y = tf.concat([x, z], axis=1)
            y = tf.layers.dense(y, 512, tf.nn.leaky_relu, name='y_1')
            y = tf.layers.dense(y, 256, tf.nn.leaky_relu, name='y_2')
            y = tf.layers.dense(y, 128, tf.nn.leaky_relu, name='y_3')
            out = tf.layers.dense(y, 1, name='y_4')
        return out

    def build_model(self, learning_rate=0.0002):

        x = tf.placeholder(tf.float32, [None, self.img_rows * self.img_cols])
        z = tf.placeholder(tf.float32, [None, self.z_dim])

        x_en = self.encoder(x)
        z_de = self.decoder(z)

        d_real = self.discriminator(x_en, x)
        d_fake = self.discriminator(z, z_de)

        # discriminator
        loss_dis_enc = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
        loss_dis_gen = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
        dis_loss = loss_dis_gen + loss_dis_enc
        # decoder
        d_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))
        # encoder
        e_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_real), logits=d_real))

        t_vars = tf.trainable_variables()
        dis_vars = [var for var in t_vars if 'discriminator' in var.name]
        d_vars = [var for var in t_vars if 'decoder' in var.name]
        e_vars = [var for var in t_vars if 'encoder' in var.name]

        # 优化器
        dis_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(dis_loss, var_list=dis_vars)
        d_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
        e_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(e_loss, var_list=e_vars)

        return x, z, \
            dis_loss, d_loss, e_loss, \
            e_optimizer, d_optimizer, dis_optimizer, \
            z_de, x_en

    def train(self, train_steps=100000, batch_size=100, learning_rate=0.0002, save_model_numbers=3):
        x, z, \
            dis_loss, d_loss, e_loss, \
            e_optimizer, d_optimizer, dis_optimizer, \
            z_de, x_en = self.build_model(learning_rate)

        saver = tf.train.Saver(max_to_keep=save_model_numbers)
        if not os.path.exists('out/'):
            os.makedirs('out/')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(train_steps):
                batch_index = np.random.randint(0, self.img_counts, batch_size)
                batch_real = self.train_images[batch_index]
                z_sample = np.random.uniform(-1., 1., size=[batch_size, self.z_dim])

                sess.run(dis_optimizer, feed_dict={x: batch_real, z: z_sample})
                sess.run(d_optimizer, feed_dict={x: batch_real, z: z_sample})
                sess.run(e_optimizer, feed_dict={x: batch_real, z: z_sample})

                if i % 1000 == 0:
                    dis_loss_curr = sess.run(dis_loss, feed_dict={x: batch_real, z: z_sample})
                    d_loss_curr = sess.run(d_loss, feed_dict={x: batch_real, z: z_sample})
                    e_loss_curr = sess.run(e_loss, feed_dict={x: batch_real, z: z_sample})

                    print('step: ' + str(i))
                    print('dis_loss: ' + str(dis_loss_curr))
                    print('d_loss:   ' + str(d_loss_curr))
                    print('e_loss:   ' + str(e_loss_curr))
                    print()
                    saver.save(sess, 'ckpt/mnist.ckpt', global_step=i)

                    r, c = 10, 10
                    # batch_fake_img = np.random.uniform(-1., 1., size=[r * c, self.noise_dim])
                    samples = sess.run(z_de, feed_dict={x: batch_real, z: z_sample})
                    fig, axs = plt.subplots(r, c)
                    cnt = 0
                    for p in range(r):
                        for q in range(c):
                            axs[p, q].imshow(np.reshape(samples[cnt], (28, 28)), cmap='gray')
                            axs[p, q].axis('off')
                            cnt += 1
                    fig.savefig("out/%d.png" % i)
                    plt.close()

                    test_z = sess.run(x_en, feed_dict={x: self.train_images})
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    ax.scatter(test_z[:, 0], test_z[:, 1], c=self.train_labels_one_dim, s=10)
                    # ax.scatter(x[:, 0], x[:, 2], x[:, 3], c=y, s=10)
                    fig.savefig("out/%d_prediction.png" % i)
                    plt.close()

    def restore_model(self):
        x, z, \
        dis_loss, d_loss, e_loss, \
        e_optimizer, d_optimizer, dis_optimizer, \
        z_de, x_en = self.build_model(0.001)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model_file = tf.train.latest_checkpoint('ckpt/')
            saver.restore(sess, model_file)
            # write your code

            # r, c = 10, 10
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
    model = BiGAN(datas[1])
    model.train()
    # model.restore_model()
