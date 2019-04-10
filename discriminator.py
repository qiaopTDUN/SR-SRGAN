import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim

batch_norm_params = {
        'decay': 0.995,
        'epsilon': 0.001,
        'updates_collections': None,
        # 'scale': True,  # [test1]
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES]}  # [test2: removed from 'trainable_variables']


def discriminator(feature,
                  weight_decay=0.05,
                  is_training=True):
    with tf.variable_scope('discriminator', 'discriminator', [feature], reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            # k3n64s1 + Leaky ReLU
            net = slim.conv2d(feature, 64, [3, 3], activation_fn=None, scope='conv2d_1')
            net = tf.nn.leaky_relu(net, 0.2, 'leaky_relu_1')

            # k3n64s2 + BN + PReLU
            net = slim.conv2d(net, 64, [3, 3], 2, activation_fn=None, scope='conv2d_2')
            net = slim.batch_norm(net, param_initializers=batch_norm_params)
            net = tf.nn.leaky_relu(net, 0.2, 'leaky_relu_2')

            # B conv blocks
            net = slim.conv2d(net, 128, [3, 3], 1, activation_fn=None, scope='conv2d_3_1')
            net = slim.batch_norm(net, param_initializers=batch_norm_params)
            net = tf.nn.leaky_relu(net, 0.2, 'leaky_relu_3_1')

            net = slim.conv2d(net, 128, [3, 3], 2, activation_fn=None, scope='conv2d_3_2')
            net = slim.batch_norm(net, param_initializers=batch_norm_params)
            net = tf.nn.leaky_relu(net, 0.2, 'leaky_relu_3_2')

            net = slim.conv2d(net, 256, [3, 3], 1, activation_fn=None, scope='conv2d_3_3')
            net = slim.batch_norm(net, param_initializers=batch_norm_params)
            net = tf.nn.leaky_relu(net, 0.2, 'leaky_relu_3_3')

            net = slim.conv2d(net, 256, [3, 3], 2, activation_fn=None, scope='conv2d_3_4')
            net = slim.batch_norm(net, param_initializers=batch_norm_params)
            net = tf.nn.leaky_relu(net, 0.2, 'leaky_relu_3_4')

            net = slim.conv2d(net, 512, [3, 3], 1, activation_fn=None, scope='conv2d_3_5')
            net = slim.batch_norm(net, param_initializers=batch_norm_params)
            net = tf.nn.leaky_relu(net, 0.2, 'leaky_relu_3_5')

            net = slim.conv2d(net, 512, [3, 3], 2, activation_fn=None, scope='conv2d_3_6')
            net = slim.batch_norm(net, param_initializers=batch_norm_params)
            net = tf.nn.leaky_relu(net, 0.2, 'leaky_relu_3_6')

            # dense + Leaky ReLU
            net = slim.conv2d(net, 1024, [1, 1], activation_fn=None, scope='conv2d_4')
            net = tf.nn.leaky_relu(net, 0.2, 'leaky_relu_4')

            # dense + sigmoid
            net = slim.conv2d(net, 1, [1, 1], activation_fn=None, scope='conv2d_5')
            # net = tf.nn.sigmoid(net, 'output')

            return net