from __future__ import print_function, division
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
import os


class VAE:
    model_name = "VAE"
    paper_name = "Auto-Encoding Variational Bayes(变分自编码器)"
    paper_url = "https://arxiv.org/abs/1312.6114"
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
            out_mean = tf.layers.dense(e, 2, name='e_4')
            out_stddev = 1e-6 + tf.layers.dense(e, 2, tf.nn.softplus, name='e_4')
            return out_mean, out_stddev

    def decoder(self, d):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            d = tf.layers.dense(d, 128, tf.nn.relu, name='d_1')
            d = tf.layers.dense(d, 256, tf.nn.relu, name='d_2')
            d = tf.layers.dense(d, 512, tf.nn.relu, name='d_3')
            out = tf.layers.dense(d, 784, name='d_4')
            return out

    def build_model(self, learning_rate=0.0002):

        x_real = tf.placeholder(tf.float32, [None, self.img_rows * self.img_cols])
        z_noise = tf.placeholder(tf.float32, [None, 2])

        z_mean, z_stddev = self.encoder(x_real)
        # 我们选择拟合logσ2而不是直接拟合σ2，是因为σ2总是非负的，
        # 需要加激活函数处理，而拟合logσ2不需要加激活函数，因为它可正可负。
        # guessed_z = z_mean + tf.exp(z_log_stddev2 / 2) * samples
        samples = tf.random_normal(tf.shape(z_stddev), 0, 1, dtype=tf.float32)
        guessed_z = z_mean + z_stddev * samples
        x_fake = self.decoder(guessed_z)

        z_real = self.decoder(z_noise)

        # marginal_likelihood = -tf.reduce_sum(
        #                           x_real * tf.log(1e-8 + x_fake) + (1 - x_real) * tf.log(1e-8 + 1 - x_fake),1)
        # 用下面的函数接口，计算方法和上述一样，当最后输出是去除sigmoid激活函数
        marginal_likelihood = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_fake, labels=x_real),
                                            reduction_indices=1)
        # # 拟合logσ2
        # kl_divergence = -0.5 * tf.reduce_sum(1 + z_stddev - tf.pow(z_mean, 2) - tf.exp(z_stddev),
        #                                      reduction_indices=1)
        # 拟合σ
        kl_divergence = 0.5 * tf.reduce_sum(
            tf.square(z_mean) + tf.square(z_stddev - 1), 1)
        # kl_divergence = 0.5 * tf.reduce_sum(
        #     tf.square(z_mean) + tf.square(z_stddev) - tf.log(1e-8 + tf.square(z_stddev)) - 1, 1)
        cost = tf.reduce_mean(marginal_likelihood + kl_divergence)

        t_vars = tf.trainable_variables()
        e_d_vars = [var for var in t_vars if 'encoder' or 'decoder' in var.name]
        optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost, var_list=e_d_vars)

        return x_real, x_fake, cost, optimizer, z_noise, z_real, guessed_z

    def train(self, train_steps=100000, batch_size=100, learning_rate=0.001, save_model_numbers=3):
        x_real, x_fake, cost, optimizer, z_noise, z_real, guessed_z = self.build_model(learning_rate)
        saver = tf.train.Saver(max_to_keep=save_model_numbers)
        if not os.path.exists('out/'):
            os.makedirs('out/')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # merged_summary_op = tf.summary.merge_all()
            # summary_writer = tf.summary.FileWriter('log/mnist_with_summaries', sess.graph)
            for i in range(train_steps):
                batch_index = np.random.randint(0, self.img_counts, batch_size)
                batch_real = self.train_images[batch_index]

                sess.run(optimizer, feed_dict={x_real: batch_real})

                if i % 1000 == 0:

                    auto_encoder_loss_curr = sess.run(cost, feed_dict={x_real: batch_real})

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

                    test_z = sess.run(guessed_z, feed_dict={x_real: self.train_images})
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    ax.scatter(test_z[:, 0], test_z[:, 1], c=self.train_labels_one_dim, s=1)
                    fig.savefig("out/%d_prediction.png" % i)
                    plt.close()



    def restore_model(self):
        x_real, x_fake, cost, optimizer, z_noise, z_real, guessed_z = self.build_model()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model_file = tf.train.latest_checkpoint('ckpt/')
            saver.restore(sess, model_file)

            test_z = sess.run(guessed_z, feed_dict={x_real: self.test_images})
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            # ax.imshow(np.reshape(test_z, (28, 28)), cmap='gray')
            ax.scatter(test_z[:, 0], test_z[:, 1], c=self.test_labels_one_dim, s=1)
            # ax.scatter(test_z[:, 0], test_z[:, 1], s=10)
            # ax.scatter(x[:, 0], x[:, 2], x[:, 3], c=y, s=10)

            plt.show()


if __name__ == '__main__':
    data = ['fashion_mnist', 'mnist']
    model = VAE(data[1])
    model.train()
    # model.restore_model()
