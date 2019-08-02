# Following https://keras.io/layers/writing-your-own-keras-layers/
# https://www.tensorflow.org/beta/guide/keras/custom_layers_and_models

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D
import numpy as np
import pdb

# Deformable 1D Convolution
class DeformableConv1D(Conv1D):
    def __init__(self, filters, kernel_size, offset, batch_size, **kwargs):
        super(DeformableConv1D, self).__init__(
                    filters=filters, 
                    kernel_size=kernel_size,
                    **kwargs)
        self.batch_size = batch_size

        self.R = tf.constant(
                    regularGrid(kernel_size), 
                    tf.float32)

        self.offset = offset
        self.offset.set_shape([self.batch_size,
                          offset.shape[1],
                          offset.shape[2]])
        self.offset = tf.reshape(self.offset, [-1, 1])

        #self.dpn = tf.reshape(
        #            offset, 
        #            shape=(-1, kernel_size))

        #self.dpn = tf.math.reduce_mean(self.dpn, [0])

        #assert(self.dpn.shape == self.R.shape)

    def build(self, input_shape):
        super(DeformableConv1D, self).build(input_shape)

    def call(self, x):
        # output feature map
        x1 = tf.identity(x)
        x1.set_shape([self.batch_size,
                          x.shape[1],
                          x.shape[2]])
        y = linearInterpolation(x1, self.R, self.offset)
        y = tf.reshape(y, tf.shape(x))
        y = super(DeformableConv1D, self).call(y)
        return y

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
    Q = tf.expand_dims(Q, [-1])
    P = Q + off
    P = tf.transpose(P)
    op = lambda p: off_update(Q, p, x1d)
    # every column will contain the corresponding g(q,p) sum
    Poff = tf.map_fn(op, P, dtype=tf.float32)
    # and the sum of every column represent each output element
    y = tf.reduce_sum(Poff, [0])
    return y

"""
    for each tensor p in P:
        for each tensor q in Q:
            update q with g(q,p)
"""
def off_update(Q, p, x):
    G = tf.map_fn(lambda q: g(q, p), Q, tf.float32)
    G = G * x
    G = tf.reduce_sum(G, [0])
    return G

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
