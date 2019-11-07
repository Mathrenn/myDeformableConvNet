import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Conv1D, Dense
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn_ops
import numpy as np
import pdb

# Deformable 1D Convolution
class DeformableConv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super(DeformableConv1D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.R = tf.constant(self.regularGrid(self.kernel_size), tf.float32)

    def build(self, input_shape):
        W_shape = (self.kernel_size, 1)
        self.W = self.add_weight(
            name='W',
            shape=W_shape,
            trainable=True,
            dtype=self.dtype)
        super(DeformableConv1D, self).build(input_shape)

    def call(self, x):
        offconv = Conv1D(x.shape[-1]*2, self.kernel_size, padding='same', activation='relu', trainable=True)
        offset = offconv(x)
        y = self.linearInterpolation(x, offset)
        y = tf.reduce_sum(self.W * y, [0])
        y = tf.reshape(y, [-1, x.shape[1], x.shape[2]])
        return y

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
        # input locations
        Q = tf.where(tf.equal(K.flatten(x), K.flatten(x)))
        Q = tf.cast(Q, tf.float32)

        offset = offset - x
        offset = K.flatten(offset)

        # offset locations
        P = Q + offset

        # regulard grid sampling
        ylist = []
        for pn in tf.unstack(self.R):
          G = self.g(Q, P+pn)
          ylist.append(G * K.flatten(x))

        return tf.stack(ylist)

    """
        linear interpolation kernel
        q: input location
        p: offset location
    """
    def g(self, q, p):
        g = tf.subtract(tf.squeeze(q), tf.squeeze(p))
        g = tf.abs(g)
        g = tf.subtract(1.0, g)
        return tf.maximum(0.0, g)
