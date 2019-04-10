import os
import scipy.misc
import numpy as np

from srresnet import srresnet
from srgan import srgan
from utils import pp, visualize, to_json, save_images
from glob import glob
import scipy.misc
from scipy.misc import imresize
import tensorflow as tf
from generator import generator_sr
import time

flags = tf.app.flags
flags.DEFINE_integer("epoch", 200, "Epoch to train [25]")
flags.DEFINE_string("optimizer", "RMSProp", "optimization methods [Adam, SGD, RMSProp]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.9]")
flags.DEFINE_float("learning_rate", 0.0001, "the initial learning  rate [0.0001]")
flags.DEFINE_float("momentum", 0.9, "Momentum term of SGD [0.9]")
flags.DEFINE_float("lr_decay_step", 800*100, "Learning rate decay step of SGD [704]")
flags.DEFINE_float("lr_decay_rate", 0.1, "Learning rate decay rate of SGD [0.1]")
flags.DEFINE_integer("train_size", 999999, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 8, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 384, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string("dataset", "DIV2K", "The name of dataset [celebA, mnist, lsun, DIV2K]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_generator", True, "True for generator, False for gan")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS


def load(checkpoint_dir, sess, saver):
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


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if FLAGS.is_train:
        with tf.Session(config=config) as sess:
            if FLAGS.is_generator:
                sr = srresnet(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                        dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir)

                sr.train(FLAGS)
            else:
                gan = srgan(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                            dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir)

                gan.train(FLAGS)
    else:
        if FLAGS.is_generator:
            start_time = time.time()
            data_lr = sorted(glob(os.path.join("./data/DIV2K/DIV2K_valid_LR_bicubic/X4", "*.png")))
            data_hr = sorted(glob(os.path.join("./data/DIV2K/valid", "*.png")))

            # data_lr = sorted(glob(os.path.join("./data/set5/set5_LR/X4", "*.png")))
            # data_hr = sorted(glob(os.path.join("./data/set5/set5_HR", "*.png")))

            # data_lr = sorted(glob(os.path.join("./data/set14/set14_LR/X4", "*.png")))
            # data_hr = sorted(glob(os.path.join("./data/set14/set14_HR", "*.png")))

            # data_lr = sorted(glob(os.path.join("./data/DIV2K/test", "*.bmp")))
            # data_hr = None

            img_idx = 64

            if data_hr is not None:
                assert len(data_hr)==len(data_lr)

            sample_files = data_lr[img_idx]
            print(sample_files)
            sample_input_images = scipy.misc.imread(sample_files).astype(np.float)
            sample_input_images = np.array(sample_input_images, dtype=np.float32)
            h, w, ch = sample_input_images.shape
            sample_input_images = np.expand_dims(sample_input_images, axis=0)

            inputs = tf.placeholder(tf.float32, [1, h, w, ch], name='real_images')
            G = generator_sr(inputs)
            t_vars = tf.trainable_variables()
            g_vars = [var for var in t_vars if 'generator' in var.name]
            saver = tf.train.Saver(g_vars)

            print('build model costs: %.6f seconds' % (time.time() - start_time))
            with tf.Session(config=config) as sess:
                #sess.run(tf.global_variables_initializer())
                start_time = time.time()
                could_load, checkpoint_counter = load("./checkpoint/DIV2K_8_1_2e-2_2e-6_1e-5_1e-4/", sess, saver)
                # could_load, checkpoint_counter = load("./models/srresnet/", sess, saver)

                if could_load:
                    print(" [*] Load SUCCESS")
                else:
                    print(" [!] Load failed...")

                print('load ckpt costs: %.6f seconds' % (time.time() - start_time))

                start_time = time.time()

                sample_output_images = sess.run([G], feed_dict={inputs: sample_input_images})
                sample_output_images = sample_output_images[0]
                sample_output_images = np.array(sample_output_images).astype(np.float32)
                sample_output_images = np.squeeze(sample_output_images, axis=0)

                tim = time.time() - start_time
                print("Evaluation: [%2d] time: %4.4f" % (img_idx, tim))

                if data_hr is not None:
                    sample_files = data_hr[img_idx]
                    print(sample_files)
                    sample_input_images = scipy.misc.imread(sample_files).astype(np.float)
                    sample_input_images = np.array(sample_input_images, dtype=np.float32) / 255.
                    diff = sample_input_images - (sample_output_images+1.)/2.
                    diff = np.reshape(diff, (1, -1))
                    rmse = np.sqrt(np.mean(diff ** 2, 1))
                    psnr = 20 * np.log10(1 / rmse)
                    print("Evaluation: [%2d] psnr: %.8f" % (img_idx, psnr))

                scipy.misc.imsave('./samples/%d-sr.png' % (img_idx), (sample_output_images+1.)/2.)
                # scipy.misc.imsave('./samples/%d-sr1.png' % (img_idx), sample_output_images)
        else:
            pass

    if FLAGS.visualize:
        # to_json("./web/js/layers.js", [sr.h0_w, sr.h0_b, sr.g_bn0],
        #                               [sr.h1_w, sr.h1_b, sr.g_bn1],
        #                               [sr.h2_w, sr.h2_b, sr.g_bn2],
        #                               [sr.h3_w, sr.h3_b, sr.g_bn3],
        #                               [sr.h4_w, sr.h4_b, None])

        # Below is codes for visualization
        OPTION = 2
        # visualize(sess, sr, FLAGS, OPTION)

if __name__ == '__main__':
    tf.app.run()