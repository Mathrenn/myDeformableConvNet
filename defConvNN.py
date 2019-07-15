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
                                               kernel_initializer='zeros',
                                               **kwargs)

    # use super (default) weights
    def build(self, input_shape):
        super(DeformableConv1D, self).build(input_shape)

    def call(self, x):
        # offset values are computed with a standard convolutional layer
        offset = super(DeformableConv1D, self).call(x)
        #assert(offset.shape[2] == x.shape[2])
        # regular grid
        R = tf.constant(regularGrid(self.kernel_size[0]))
        dpn = tf.math.reduce_mean(offset, [0,-1])
        # output feature map
        y = linearInterpolation(x, R, dpn)
        y = tf.reshape(y, tf.shape(x))
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
    y = tf.numpy_function(linInterOp, [x, R, offset], tf.float32)
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
    y = np.zeros_like(x)
    for p0 in range(y.shape[0]):
        for pn in R:
            # offset location
            p = p0 + pn + dpn[pn]
            glist = np.zeros_like(x)
            for q in range(x.shape[0]):
                glist[q] = G(q,p)
            y[p0] = np.sum(x * glist)
    return y

"""
    linear interpolation kernel
    
    q: integer, input location
    p: integer, offset location
"""
def G(q, p):
    return max(0, 1-(abs(q-p)))
