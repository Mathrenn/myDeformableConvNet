# Following https://keras.io/layers/writing-your-own-keras-layers/
# https://www.tensorflow.org/beta/guide/keras/custom_layers_and_models

import tensorflow as tf
import numpy as np
import pdb
from tensorflow.keras.layers import Layer, Conv1D

# Deformable 1D Convolution
class DeformableConv1D(Conv1D):
    def __init__(self, filters, kernel_size, **kwargs):
        super(DeformableConv1D, self).__init__(filters=filters, 
                                               kernel_size=kernel_size,
                                               padding='CAUSAL',
                                               **kwargs)

    # use super (default) weights
    def build(self, input_shape):
        super(DeformableConv1D, self).build(input_shape)

    def call(self, x):
        # offset values are computed with a standard convolutional layer
        offset = super(DeformableConv1D, self).call(x)
        # regular grid
        R = tf.constant(regularGrid(self.kernel_size[0]), tf.float32)
        dpn = tf.reshape(offset, (-1, self.kernel_size[0]))
        dpn = tf.math.reduce_mean(dpn, [0])
        assert(dpn.shape == R.shape)
        # output feature map
        y = linearInterpolation(x, R, dpn)
        y = tf.reshape(y, tf.shape(x))
        y = super(DeformableConv1D, self).call(y)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

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
    x_reshaped = tf.reshape(x, [-1])
    xshape = tf.shape(x_reshaped).get_shape()
    Q = tf.range(xshape.as_list()[0], dtype=tf.float32)
    Q = tf.expand_dims(Q, [-1])
    P = Q + off
    P = tf.transpose(P)
    # every column will contain the corresponding g(q,p) sum
    Poff = tf.map_fn(lambda p: off_update(Q, p, x_reshaped), 
                P, 
                dtype=tf.float32)
    # and the sum of every column represent each output element
    y = tf.reduce_sum(Poff, [0])
    return y

"""
    for each tensor p in P:
        for each tensor q in Q:
            update q with g(q,p)
"""
def off_update(Q, p, x):
    G = tf.identity(Q)
    G = tf.map_fn(lambda q: g(q, p), G, tf.float32)
    G = G * x
    return tf.reduce_sum(G, [0])

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
