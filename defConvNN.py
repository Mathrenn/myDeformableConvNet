# Following https://keras.io/layers/writing-your-own-keras-layers/
# https://www.tensorflow.org/beta/guide/keras/custom_layers_and_models

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn_ops
import numpy as np
import pdb

# Deformable 1D Convolution
class DeformableConv1D(Conv1D):
    def __init__(self, filters, kernel_size, offset, **kwargs):
        super(DeformableConv1D, self).__init__(
                    filters=filters, 
                    kernel_size=kernel_size,
                    **kwargs)

        self.offshape = offset.shape
        self.offset = tf.reshape(offset, [-1, 1])

        self.R = tf.constant(
                    regularGrid(kernel_size), 
                    tf.float32)


    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=False,
            dtype=self.dtype)
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()

        self.built = True

    def call(self, x):
        # output feature map
        y = linearInterpolation(x, self.R, self.offset)
        y = tf.reshape(y, self.offshape)
        return y

    def compute_output_shape(self, input_shape):
        return super(DeformableConv1D, self).compute_output_shape(input_shape)

"""
    Regular grid

    kernel_size: integer
"""
def regularGrid(kernel_size):
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
def linearInterpolation(x, R, offset):
    # output map
    y = linInterOp(x, R, offset)
    return y

"""
    linear interpolation operation

    for each location p0 in the output map y
     for each location pn in the regular grid R
      x(p) = sum(g(q, p), x(q)) over the input locations q
       with p = p0 + pn + dpn and dpn offset location
    
    input:
        x: input feature map
        R: regular grid
        dpn: offsets
    output:
        y: offset feature map
"""
def linInterOp(x, R, dpn):
    # TODO: make this work on tensors with shape None
    R = tf.cast(R, tf.float32)
    off = dpn + R
    off = tf.math.reduce_mean(off, [0])
    x1d = tf.reshape(x, [-1])
    xshape = x1d.shape

    Q = tf.range(xshape[0])
    Q = tf.cast(Q, tf.float32)

    P = tf.range(tf.shape(tf.reshape(dpn, [-1]))[0])
    P = tf.cast(P, tf.float32)

    y = tf.map_fn(lambda p: G(Q, p, x1d), P)
    return y

def G(Q, p, x):
    P = tf.fill(Q.shape, p)
    P = g(Q, P)
    P = P * x 
    P = tf.reduce_sum(P)
    return P

"""
    linear interpolation kernel
    
    q: input location
    p: offset location
"""
def g(q, p):
    g = q-p
    g = tf.abs(g)
    g = tf.subtract(1.0, g)
    return tf.maximum(0.0, g)
