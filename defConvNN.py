# Following https://keras.io/layers/writing-your-own-keras-layers/
# https://www.tensorflow.org/beta/guide/keras/custom_layers_and_models

from keras.layers import Layer, Conv1D
import tensorflow as tf
import numpy as np
import pdb

# Deformable 1D Convolution
class DeformableConv1D(Conv1D):
    def __init__(self, filters, kernel_size,**kwargs):
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
        assert(offset.shape[2] == x.shape[2])
        # regular grid
        R = regularGrid(self.kernel_size)
        # output feature map
        y = linearInterpolation(x, offset, R)
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
def linearInterpolation(x, offset, R):
    # output map
    y = tf.zeros_like(x)

    # for each location p0 in the output map y
    ## for each location pn in the regular grid R
    ### x(p) = sum(g(q, p), x(q)) over the input locations q
    ### with p = p0 + pn + dpn and dpn offset location

"""
    linear interpolation kernel
    
    q: integer, input location
    p: integer, offset location
"""
def G(q, p):
    return max(0, 1-(abs(q-p)))
