#!pip install tensorflow-gpu==2.0.0-rc2

# Following https://keras.io/layers/writing-your-own-keras-layers/
# https://www.tensorflow.org/beta/guide/keras/custom_layers_and_models

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Conv1D
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn_ops
import numpy as np
import pdb

# Deformable 1D Convolution
class DeformableConv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(DeformableConv1D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.R = tf.constant(self.regularGrid(self.kernel_size), tf.float32)
        self.offconv = Conv1D(filters, kernel_size, trainable=True, **kwargs)

    def build(self, input_shape):
        W_shape = (self.kernel_size, 1)
        self.W = self.add_weight(
            name='W',
            shape=W_shape,
            trainable=True,
            dtype=self.dtype)
        self.built = True

    def call(self, x):
        # output feature map
        offset = self.offconv(x)
        y = self.linearInterpolation(x, offset)
        y = tf.reduce_sum(self.W * y, [0])
        
        #y = tf.reshape(y, [-1, offset.shape[1], offset.shape[2]])
        y = tf.reshape(y, [-1, x.shape[1], x.shape[2]])
        
        return self.offconv(y)

    """
       Regular grid
       kernel_size: integer
    """
    def regularGrid(self, kernel_size):
        R = np.zeros(kernel_size, dtype='int32')
        j = -(np.floor(kernel_size/2))
        for i in range(R.shape[0]):
            R[i] = j
            j += 1
        return R

    """
        linear interpolation
        x: (b, ts, c)
        offset: (b, ts, c)
        R: (kernel_size)
    """
    def linearInterpolation(self, x, offset):
        Q = tf.where(tf.equal(K.flatten(x), K.flatten(x)))
        Q = tf.cast(Q, tf.float32)
        #Q = tf.range(0, x.shape[0]*x.shape[1]*x.shape[2], dtype=tf.float32)

        R = tf.cast(self.R, tf.float32)
        dpn = tf.reshape(offset, [-1, 1]) + R
        dpn = tf.math.reduce_mean(dpn, [0])

        dpn_list = tf.unstack(dpn)
        ylist = []

        for pn in dpn_list:
          G = self.g(Q, Q+pn)
          ylist.append(G * K.flatten(x))

        return tf.stack(ylist)

    """
        linear interpolation kernel
        q: input location
        p: offset location
    """
    def g(self, q, p):
        #pdb.set_trace()
        g = tf.subtract(tf.squeeze(q), tf.squeeze(p))
        g = tf.abs(g)
        g = tf.subtract(1.0, g)
        return tf.maximum(0.0, g)

