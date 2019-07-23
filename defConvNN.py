# Following https://keras.io/layers/writing-your-own-keras-layers/
# https://www.tensorflow.org/beta/guide/keras/custom_layers_and_models

from keras.layers import Layer, Conv1D
import tensorflow as tf
import numpy as np
import pdb

# Deformable 1D Convolution
class DeformableConv1D(Conv1D):
    def __init__(self, filters, kernel_size, **kwargs):
        super(DeformableConv1D, self).__init__(filters=filters, 
                                               kernel_size=kernel_size,
                                               **kwargs)

    # use super (default) weights
    def build(self, input_shape):
        super(DeformableConv1D, self).build(input_shape)

    def call(self, x):
        # offset values are computed with a standard convolutional layer
        offset = super(DeformableConv1D, self).call(x)
        # regular grid
        R = regularGrid(self.kernel_size[0])
        # TODO: find appropriate reduce function
        dpn = tf.reshape(offset, (-1, self.kernel_size[0]))
        dpn = tf.math.reduce_mean(dpn, [0])
        assert(dpn.shape == R.shape)
        # output feature map
        y = linearInterpolation(x, R, dpn)
        y = tf.reshape(y, tf.shape(x))
        return super(DeformableConv1D, self).call(y)

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
    #y = tf.numpy_function(linInterOp, [x, R, offset], tf.float32)
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
    y = tf.zeros_like(x)
    y = tf.Variable(y)
    for p0 in range(y.shape[0]):
        for pn in R:
            # offset locations
            P = dpn + pn + p0
            Q = tf.cast(tf.range(tf.reshape(x, [-1]).shape[0]), tf.float32)
            G = tf.Variable(off_update(Q, P, x))
            # update y(p0) with the last value
            tf.compat.v1.scatter_add(y, [p0], G)
    return y

"""
    for each tensor p in P:
        for each tensor q in Q:
            update q with g(q,p)
"""
def off_update(Q, P, x):
    Qlist = tf.zeros_like(Q)
    Qlist = tf.Variable(Qlist)
    for p in tf.unstack(tf.reshape(P, [-1])):
        for q in tf.unstack(tf.reshape(Q, [-1])):
            tf.scatter_update(Qlist, tf.where(tf.equal(Qlist, q)), g(q, p))
    # NOTE: reduce_mean returns values most similar to a standard conv
    return tf.reduce_mean(tf.stack(Qlist) * tf.reshape(x, [-1]), [0])

"""
    linear interpolation kernel
    
    q: input location
    p: offset location
"""
def g(q, p):
    return tf.maximum(0, tf.subtract( 1, (tf.abs(q-p))))
