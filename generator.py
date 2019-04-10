import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim

batch_norm_params = {
        'decay': 0.995,
        'epsilon': 0.001,
        'updates_collections': None,
        # 'scale': True,  # [test1]
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES]}  # [test2: removed from 'trainable_variables']


def _phase_shift(I, r, scope=None):
    with tf.variable_scope(scope):
        bsize, a, b, c = I.get_shape().as_list()
        bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
        X = tf.reshape(I, (bsize, a, b, r, r))
        X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r
        return tf.reshape(X, (bsize, a*r, b*r, 1))


def PS(X, r, color=False, scope=None):
    with tf.variable_scope(scope):
        if color:
            Xc = tf.split(X, 64, 3)
            cnt = 0
            cX = []
            for x in Xc:
                op_name = 'sp_{}'.format(cnt)
                cX.append(_phase_shift(x, r, op_name))
                cnt += 1

            X = tf.concat(cX, 3)
            # X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
        else:
            X = _phase_shift(X, r)
        return X


def prelu(feature, scope='prelu'):
    with tf.variable_scope(scope):
        alphas = tf.get_variable(scope+'_alpha', feature.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        pos = tf.nn.relu(feature)
        neg = alphas * (feature - tf.abs(feature)) * 0.5
        return pos + neg


def res_block(feature, kern_sz=3, channel_num=64, stride=1, weight_decay=0.05, scope=None):
    with tf.variable_scope(scope):
        net = slim.conv2d(feature, channel_num, [kern_sz, kern_sz], stride,
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          activation_fn=None)
        net = slim.batch_norm(net, param_initializers=batch_norm_params)
        net = prelu(net, scope)
        net = slim.conv2d(net, channel_num, [kern_sz, kern_sz], stride,
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          activation_fn=None)
        net = slim.batch_norm(net, param_initializers=batch_norm_params)
        net = net + feature
        return net


def generator_sr(feature,
              weight_decay=0.05,
              up_scale=4,
              is_training=True):

    with tf.variable_scope('generator'):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            # k9n64s1 + PReLU
            net = slim.conv2d(feature, 64, [9, 9], activation_fn=None, scope='conv2d_1')
            net = prelu(net, 'prelu_1')

            # B residual blocks
            # k3n64s1 + BN + PReLU + k3n64s1 + BN
            resnet = net
            for blk_i in range(16):
                resnet = res_block(resnet, 3, 64, 1, weight_decay, 'resblock_{}'.format(blk_i))

            # k3n64s1 + BN
            resnet = slim.conv2d(resnet, 64, [3, 3], activation_fn=None, scope='conv2d_2')
            resnet = slim.batch_norm(resnet, param_initializers=batch_norm_params)
            net = net + resnet

            # subpixel
            spnet = slim.conv2d(net, 256, [3, 3], activation_fn=None, scope='con2d_3_1')
            spnet = PS(spnet, 2, True, 'subpixel_3_1')
            spnet = prelu(spnet, 'prelu_3_1')

            spnet = slim.conv2d(spnet, 256, [3, 3], activation_fn=None, scope='con2d_3_2')
            spnet = PS(spnet, 2, True, 'subpixel_3_2')
            spnet = prelu(spnet, 'prelu_3_2')

            # k9n3s1
            net = slim.conv2d(spnet, 3, [9, 9], activation_fn=tf.nn.tanh, scope='conv2d_4')

            return net