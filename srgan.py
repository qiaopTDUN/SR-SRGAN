import os
import time
import math
from glob import glob
import tensorflow as tf
from tensorflow.contrib import slim
from simple_vgg19_api import vgg_19

import numpy as np
from scipy.misc import imresize
from generator import generator_sr
from discriminator import discriminator
from utils import *


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def doresize(x, shape):
    x = np.copy((x+1.)*127.5).astype("uint8")
    y = imresize(x, shape)
    return y

class srgan(object):
    def __init__(self, sess, image_size=128, is_crop=True,
                 batch_size=64, image_shape=[128, 128, 3],
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir=None):
        """
        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          y_dim: (optional) Dimension of dim for y. [None]
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.input_size = 96
        self.sample_size = batch_size
        self.image_shape = [image_size, image_size, 3]

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = 3

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        else:
            self.y = None

        # generator
        # input LR image W x H,
        # output upscale = r SR image rW x rH
        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.input_size, self.input_size, 3], name='real_inputs')
        # discriminator
        # input HR image rW x rH
        # output fake or not
        self.images = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, name='real_images')
        # discriminator
        # input SR image rW x rH
        # output fake or not
        self.sample_images = tf.placeholder(tf.float32, [self.sample_size] + self.image_shape, name='sample_images')

        self.G = generator_sr(self.inputs)
        self.D_real = discriminator(self.images)
        self.D_fake = discriminator(self.G)

        ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
        means = np.array((123.68, 116.78, 103.94), dtype=np.float32)
        with tf.variable_scope('vgg_real_input'):
            self.target_image_224 = tf.image.resize_images((self.images+1.)/2.*255., size=[224, 224], method=0, align_corners=False)
            num_channels = self.target_image_224.get_shape().as_list()[-1]
            channels = tf.split(axis=3, num_or_size_splits=num_channels, value=self.target_image_224)
            for i in range(num_channels):
                channels[i] -= means[i]
            self.target_image_224 = tf.concat(axis=3, values=channels)

        with tf.variable_scope('vgg_fake_input'):
            self.predict_image_224 = tf.image.resize_images((self.G+1.)/2.*255., size=[224, 224], method=0, align_corners=False)
            num_channels = self.predict_image_224.get_shape().as_list()[-1]
            channels = tf.split(axis=3, num_or_size_splits=num_channels, value=self.predict_image_224)
            for i in range(num_channels):
                channels[i] -= means[i]
            self.predict_image_224 = tf.concat(axis=3, values=channels)

        _, self.VGG_real, _ = vgg_19(self.target_image_224, is_training=False, reuse=False)
        _, self.VGG_fake, _ = vgg_19(self.predict_image_224, is_training=False, reuse=True)

        self.d_loss_real = tf.reduce_mean(self.D_real)
        self.d_loss_fake = tf.reduce_mean(self.D_fake)
        self.d_loss = self.d_loss_real - self.d_loss_fake

        self.g_gan_loss = 1e-3 * tf.reduce_mean(self.D_fake)
        self.mse_loss = tf.reduce_mean(tf.square(self.images - self.G), name='mse')
        self.vgg_loss = 2e-6 * tf.reduce_mean(tf.square(self.VGG_real - self.VGG_fake), name='vgg')
        self.g_loss = self.g_gan_loss + self.mse_loss + self.vgg_loss

        self.real_sum = tf.summary.image("real", self.images)
        self.recover_sum = tf.summary.image("recover", self.G)
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.g_gan_loss_sum = tf.summary.scalar("g_gan_loss", self.g_gan_loss)
        self.vgg_loss_sum = tf.summary.scalar("vgg_loss", self.vgg_loss)
        self.mse_loss_sum = tf.summary.scalar("mse_loss", self.mse_loss)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        global_step = tf.train.create_global_step()
        lr = tf.train.exponential_decay(config.learning_rate, global_step, decay_steps=config.lr_decay_step,
                                        decay_rate=config.lr_decay_rate)
        if config.optimizer == 'SGD':
            d_optim = tf.train.MomentumOptimizer(learning_rate=lr, momentum=config.momentum).minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.MomentumOptimizer(learning_rate=lr, momentum=config.momentum).minimize(self.g_loss, var_list=self.g_vars)
        elif config.optimizer == 'Adam':
            d_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
        elif config.optimizer == 'RMSProp':
            d_optim = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(self.g_loss, var_list=self.g_vars)
            clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.d_vars]

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver(max_to_keep=60)

        self.g_sum = tf.summary.merge([self.recover_sum, self.mse_loss_sum, self.g_gan_loss_sum, self.vgg_loss_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.real_sum, self.d_loss_real_sum, self.d_loss_fake_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        data = sorted(glob(os.path.join("./data", config.dataset, "valid", "*.png")))
        sample_files = data[0:self.sample_size]
        sample = [get_image_samp(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_inputs = [doresize(xx, [self.input_size, ] * 2) for xx in sample]
        sample_images = np.array(sample).astype(np.float32)
        sample_input_images = np.array(sample_inputs).astype(np.float32)

        save_images(sample_input_images, [int(self.batch_size/8), 8], './samples/inputs_small.png')
        save_images(sample_images, [int(self.batch_size/8), 8], './samples/reference.png')

        counter = 1
        start_time = time.time()

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SRGAN SUCCESS")
        else:
            print(" [!] Load SRGAN failed...")

            # load srresnet
            t_vars = tf.trainable_variables()
            g_vars = [var for var in t_vars if 'generator' in var.name]
            g_saver = tf.train.Saver(g_vars)
            could_load, checkpoint_counter = self.load_srresnet("./models/srresnet/", self.sess, g_saver)
            if could_load:
                print(" [*] Load SRResNet SUCCESS")
            else:
                print(" [!] Load SRResNet failed...")

            # load VGG
            g_vars = [var for var in t_vars if 'vgg' in var.name]
            vgg_saver = tf.train.Saver(g_vars)
            could_load = self.load_vgg_19("./models/vgg_19.ckpt", self.sess, vgg_saver)
            if could_load:
                print(" [*] Load VGG SUCCESS")
            else:
                print(" [!] Load VGG failed...")

        # we only save the validation inputs once
        have_saved_inputs = False

        for epoch in range(config.epoch):
            print('epoch : {}'.format(epoch))
            data = glob(os.path.join("./data", config.dataset, "train", "*.png"))
            np.random.shuffle(data)
            batch_idxs = min(len(data), config.train_size)

            for idx in range(0, int(batch_idxs)):
                batch_file = data[idx]
                batch = get_image(batch_file, self.image_size, config.batch_size, is_crop=self.is_crop)
                input_batch = [doresize(xx, [self.input_size, ] * 2) for xx in batch]
                batch_images = np.array(batch).astype(np.float32)
                batch_inputs = np.array(input_batch).astype(np.float32)

                # Update D network
                _, summary_str, _, errD = self.sess.run([d_optim, self.d_sum, clip_D, self.d_loss], feed_dict={self.inputs: batch_inputs, self.images: batch_images})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str, errG = self.sess.run([g_optim, self.g_sum, self.g_loss], feed_dict={self.inputs: batch_inputs, self.images: batch_images})
                self.writer.add_summary(summary_str, counter)

                # # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                # _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.inputs: batch_inputs, self.images: batch_images})
                # self.writer.add_summary(summary_str, counter)

                counter += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, config.epoch, idx, batch_idxs, time.time() - start_time, errD, errG))

                if np.mod(counter, 500) == 1:
                    samples, g_loss = self.sess.run(
                        [self.G, self.g_loss],
                        feed_dict={self.inputs: sample_input_images, self.images: sample_images}
                    )

                    diff = (samples - sample_images)/2.
                    diff = np.reshape(diff, (self.batch_size, -1))
                    rmse = np.sqrt(np.mean(diff ** 2, 1))
                    psnr = 20 * np.log10(1 / rmse)

                    save_images(samples, [int(self.batch_size/8), 8],
                                './samples/d_valid_%s_%s.png' % (epoch, idx))
                    print("[Sample] g_loss: %.8f, PSNR: %.8f" % (g_loss, np.mean(psnr)))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "srwgan.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def load_srresnet(self, checkpoint_dir, sess, saver):
        import re
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def load_vgg_19(self, checkpoint_name, sess, saver):
        # import os
        # from tensorflow.python import pywrap_tensorflow
        #
        # # Read data from checkpoint file
        # reader = pywrap_tensorflow.NewCheckpointReader("./models/vgg_19.ckpt")
        # var_to_shape_map = reader.get_variable_to_shape_map()
        # # Print tensor name and values
        # for key in var_to_shape_map:
        #     print("tensor_name: ", key)
        #     print(reader.get_tensor(key))
        ckpt = tf.train.checkpoint_exists(checkpoint_name)
        if ckpt:
            saver.restore(sess, checkpoint_name)
            print(" [*] Success to read {}".format(checkpoint_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False
