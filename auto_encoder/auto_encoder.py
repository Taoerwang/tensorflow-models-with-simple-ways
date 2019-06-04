from __future__ import print_function, division
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
import os


class AutoEncoder:
    model_name = "Auto Encoder"
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

    def encoder(self, e):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            e = tf.layers.dense(e, 1024, tf.nn.relu, name='e_1')
            e = tf.layers.dense(e, 512, tf.nn.relu, name='e_2')
            e = tf.layers.dense(e, 256, tf.nn.relu, name='e_3')
            out = tf.layers.dense(e, 2, name='e_4')
            return out

    def decoder(self, d):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            d = tf.layers.dense(d, 128, tf.nn.relu, name='d_1')
            d = tf.layers.dense(d, 256, tf.nn.relu, name='d_2')
            d = tf.layers.dense(d, 512, tf.nn.relu, name='d_3')
            out = tf.layers.dense(d, 784, tf.nn.sigmoid, name='d_4')
            return out

    def build_model(self, learning_rate=0.0002):

        x_real = tf.placeholder(tf.float32, [None, self.img_rows * self.img_cols])
        # x_real = tf.placeholder(tf.float32, [None, self.noise_dim])

        z = self.encoder(x_real)
        x_fake = self.decoder(z)

        t_vars = tf.trainable_variables()
        e_d_vars = [var for var in t_vars if 'encoder'or 'decoder' in var.name]
        # d_vars = [var for var in t_vars if 'decoder' in var.name]

        # loss = tf.nn.l2_loss(x - x_fake)  # L2 loss
        autoencoder_loss = tf.reduce_mean(tf.squared_difference(x_real, x_fake))

        e_d_optimizer = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(loss=autoencoder_loss, var_list=e_d_vars)
        return x_real, z, x_fake, autoencoder_loss, e_d_optimizer

    def train(self, train_steps=100000, batch_size=100, learning_rate=0.001, save_model_numbers=3):
        x_real, z, x_fake, auto_encoder_loss, e_d_optimizer = self.build_model(learning_rate)
        saver = tf.train.Saver(max_to_keep=save_model_numbers)
        if not os.path.exists('out/'):
            os.makedirs('out/')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(train_steps):
                batch_index = np.random.randint(0, self.img_counts, batch_size)
                batch_real = self.train_images[batch_index]
                batch_labels = self.train_labels_one_dim[batch_index]

                sess.run(e_d_optimizer, feed_dict={x_real: batch_real})

                if i % 1000 == 0:

                    auto_encoder_loss_curr = sess.run(auto_encoder_loss, feed_dict={x_real: batch_real})

                    print('step: ' + str(i))
                    print('D_loss: ' + str(auto_encoder_loss_curr))
                    print()
                    saver.save(sess, 'ckpt/mnist.ckpt', global_step=i)

                    x_fake_ = sess.run(x_fake, feed_dict={x_real: batch_real})

                    r, c = 10, 10
                    fig, axs = plt.subplots(r, c)
                    cnt = 0
                    for p in range(r):
                        for q in range(c):
                            axs[p, q].imshow(np.reshape(batch_real[cnt], (28, 28)), cmap='gray')
                            axs[p, q].axis('off')
                            cnt += 1
                    fig.savefig("out/%d_real.png" % i)
                    plt.close()

                    r, c = 10, 10
                    fig, axs = plt.subplots(r, c)
                    cnt = 0
                    for p in range(r):
                        for q in range(c):
                            axs[p, q].imshow(np.reshape(x_fake_[cnt], (28, 28)), cmap='gray')
                            axs[p, q].axis('off')
                            cnt += 1
                    fig.savefig("out/%d_fake.png" % i)
                    plt.close()

                    test_z = sess.run(z, feed_dict={x_real: self.test_images})
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    ax.scatter(test_z[:, 0], test_z[:, 1], c=self.test_labels_one_dim, s=10)
                    fig.savefig("out/%d_prediction.png" % i)
                    plt.close()


    def restore_model(self):
        x_real, z, x_fake, auto_encoder_loss, e_d_optimizer = self.build_model()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model_file = tf.train.latest_checkpoint('ckpt/')
            saver.restore(sess, model_file)

            test_z = sess.run(z, feed_dict={x_real: self.test_images})
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(test_z[:, 0], test_z[:, 1], c=self.test_labels_one_dim, s=10)
            plt.show()


if __name__ == '__main__':
    datas = ['fashion_mnist', 'mnist']
    model = AutoEncoder(datas[1])
    # model.train()
    model.restore_model()
