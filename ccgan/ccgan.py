from __future__ import print_function, division
import tensorflow as tf
import tensorlayer as tl
from tensorflow import keras
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
import os


class CCGAN:
    model_name = "CCGAN(具有上下文条件生成对抗网络的半监督学习)"
    paper_name = "Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks"
    paper_url = "https://arxiv.org/abs/1611.06430"
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

    def mask_randomly(self, imgs):
        imgs = np.reshape(imgs, [-1, 28, 28])
        y1 = np.random.randint(0, 28 - 10, imgs.shape[0])
        y2 = y1 + 10
        x1 = np.random.randint(0, 28 - 10, imgs.shape[0])
        x2 = x1 + 10

        masked_imgs = np.empty_like(imgs)
        for i, img in enumerate(imgs):
            masked_img = img.copy()
            _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i],
            masked_img[_y1:_y2, _x1:_x2] = 0
            masked_imgs[i] = masked_img
        masked_imgs = np.reshape(masked_imgs, [-1, 28 * 28])
        return masked_imgs

    def discriminator(self, inputs):
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            x = tf.reshape(inputs, [-1, 784])

            x = tf.layers.dense(x, 512, tf.nn.relu)
            x = tf.layers.batch_normalization(x, momentum=0.8)
            x = tf.layers.dense(x, 512, tf.nn.leaky_relu)
            x = tf.layers.batch_normalization(x, momentum=0.8)
            x = tf.layers.dense(x, 256, tf.nn.leaky_relu)
            x = tf.layers.batch_normalization(x, momentum=0.8)
            x = tf.layers.dense(x, 128, tf.nn.leaky_relu)
            d_class = tf.layers.dense(x, 10, tf.nn.sigmoid, name='x_4_10')
            out = tf.layers.dense(x, 1, name='x_4_1')
        return d_class, out

    def generator(self, inputs, is_train=True, reuse=False):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            inputs = tf.reshape(inputs, [-1, 28, 28, 1])
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
            outputs = tf.layers.batch_normalization(pool2, training=is_train)
            outputs = tf.layers.conv2d_transpose(outputs, 64, [3, 3], strides=(2, 2), padding='SAME')
            outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=is_train), name='outputs_2')
            outputs = tf.layers.conv2d_transpose(outputs, 1, [3, 3], strides=(2, 2), padding='SAME')
            out = tf.tanh(outputs, name='outputs')
        return out

    def build_model(self, learning_rate=0.0002):

        # x = tf.placeholder(tf.float32, [None, self.img_rows, self.img_cols, self.img_dim])
        x = tf.placeholder(tf.float32, [None, 784])
        x_label = tf.placeholder(tf.float32, [None, 10])

        dis = tf.placeholder(tf.float32, [None, 784])
        dis_label = tf.placeholder(tf.float32, [None, 10])

        g_sample = self.generator(x)

        d_real_class, d_real_logits = self.discriminator(dis)
        d_fake_class, d_fake_logits = self.discriminator(g_sample)

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits, labels=tf.ones_like(d_real_logits)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits)))

        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.ones_like(d_fake_logits)))

        c_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_real_class, labels=dis_label))
        c_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_fake_class, labels=x_label))

        d_loss = d_loss + c_real_loss
        g_loss = g_loss + c_fake_loss

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]
        # 优化器
        d_optimizer = tf.train.AdamOptimizer(learning_rate, 0.5).minimize(d_loss, var_list=d_vars)
        g_optimizer = tf.train.AdamOptimizer(learning_rate, 0.5).minimize(g_loss, var_list=g_vars)
        return x, dis, x_label, dis_label, \
               d_loss, g_loss, \
               d_optimizer, g_optimizer, g_sample

    def train(self, train_steps=20000, batch_size=100, learning_rate=0.0002, save_model_numbers=3):
        x, dis, x_label, dis_label, \
        d_loss, g_loss, \
        d_optimizer, g_optimizer, \
        g_sample = self.build_model(learning_rate)
        saver = tf.train.Saver(max_to_keep=save_model_numbers)

        if not os.path.exists('out/'):
            os.makedirs('out/')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(train_steps):
                mis_index = np.random.randint(0, self.img_counts, batch_size)
                real_index = np.random.randint(0, self.img_counts, batch_size)

                mis_img = self.mask_randomly(self.train_images[mis_index])
                mis_label = self.train_labels[mis_index]

                real_img = np.reshape(self.train_images[real_index], (-1, 784))
                real_label = self.train_labels[real_index]

                sess.run(d_optimizer, feed_dict={x: mis_img, x_label: mis_label, dis: real_img, dis_label: real_label})
                sess.run(g_optimizer, feed_dict={x: mis_img, x_label: mis_label, dis: real_img, dis_label: real_label})

                if i % 1000 == 0:
                    d_loss_curr = sess.run(d_loss,
                                           feed_dict={x: mis_img, x_label: mis_label,
                                                      dis: real_img, dis_label: real_label})
                    g_loss_curr = sess.run(g_loss,
                                           feed_dict={x: mis_img, x_label: mis_label,
                                                      dis: real_img, dis_label: real_label})
                    print('step: ' + str(i))
                    print('D_loss: ' + str(d_loss_curr))
                    print('G_loss: ' + str(g_loss_curr))
                    print()
                    saver.save(sess, 'ckpt/mnist.ckpt', global_step=i)

                    test_index = np.random.randint(0, self.img_counts, batch_size)
                    test_img = self.train_images[mis_index]
                    test_mis_img = self.mask_randomly(test_img)

                    r, c = 10, 10
                    samples = sess.run(g_sample, feed_dict={x: test_mis_img})
                    fig, axs = plt.subplots(r, c)
                    cnt = 0
                    for p in range(r):
                        for q in range(c):
                            axs[p, q].imshow(np.reshape(samples[cnt], (28, 28)), cmap='gray')
                            axs[p, q].axis('off')
                            cnt += 1
                    fig.savefig("out/%d_rebuild.png" % i)
                    plt.close()

                    r, c = 10, 10
                    samples = test_mis_img
                    fig, axs = plt.subplots(r, c)
                    cnt = 0
                    for p in range(r):
                        for q in range(c):
                            axs[p, q].imshow(np.reshape(samples[cnt], (28, 28)), cmap='gray')
                            axs[p, q].axis('off')
                            cnt += 1
                    fig.savefig("out/%d_mis.png" % i)
                    plt.close()

                    r, c = 10, 10
                    samples = test_img
                    fig, axs = plt.subplots(r, c)
                    cnt = 0
                    for p in range(r):
                        for q in range(c):
                            axs[p, q].imshow(np.reshape(samples[cnt], (28, 28)), cmap='gray')
                            axs[p, q].axis('off')
                            cnt += 1
                    fig.savefig("out/%d_real.png" % i)
                    plt.close()


    def restore_model(self):
        x, dis, x_label, dis_label, \
        d_loss, g_loss, \
        d_optimizer, g_optimizer, \
        g_sample = self.build_model(0.001)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model_file = tf.train.latest_checkpoint('ckpt/')
            saver.restore(sess, model_file)
        # write your code
        #     r, c = 5, 5
        #     batch_fake_img = np.random.uniform(-1., 1., size=[r * c, self.noise_dim])
        #     samples = sess.run(g_sample, feed_dict={g: batch_fake_img})
        #     fig, axs = plt.subplots(r, c)
        #     cnt = 0
        #     for p in range(r):
        #         for q in range(c):
        #             axs[p, q].imshow(np.reshape(samples[cnt], (28, 28)), cmap='gray')
        #             axs[p, q].axis('off')
        #             cnt += 1
        #     plt.show()


if __name__ == '__main__':
    datas = ['fashion_mnist', 'mnist']
    model = CCGAN(datas[1])
    model.train()
    # model.restore_model()
