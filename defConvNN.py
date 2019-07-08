# Following https://keras.io/layers/writing-your-own-keras-layers/
# Inspired by https://github.com/lubinBoooos/DeformableConv/

from keras import backend as K
from keras.layers import Layer, Conv1D
import tensorflow as tf
import numpy as np
import pdb

# Deformable 1D Convolution
class DeformableConv1D(Conv1D):
    def __init__(self, filters, **kwargs):
        super(DeformableConv1D, self).__init__(filters=2*filters, kernel_size=7,
                                               dilation_rate=1, padding='same',
                                               use_bias=False,
                                               kernel_initializer='zeros',
                                               **kwargs)

    def build(self, input_shape):
        super(DeformableConv1D, self).build(input_shape)

    def call(self, x):
        # offset values are computed with a standard convolutional layer
        offset = super(DeformableConv1D, self).call(x)
        assert(offset.shape[2] == 2*x.shape[2])
        # output feature map
        y = linearInterpolation(x, offset)
        y = tf.reshape(y, tf.shape(x))
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

"""
    linear interpolation

    x: (b, ts, c)
    offset: (b, ts, 2*c)
"""
def linearInterpolation(x, offset):
    b = tf.shape(x)[0]
    ts = tf.shape(x)[1]
    c = tf.shape(x)[2]

    # offset: (b, ts, 2*c) -> (b*c, ts, 2)
    offset = tf.transpose(offset, (0, 2, 1))
    offset = tf.reshape(offset, (b*c, 2, ts))
    offset = tf.transpose(offset, (0, 2, 1))

    # rebuild the input vector
    # vec: (ts, 2)
    vec = tf.stack([(ts, 1), (ts, 1)], axis=-1)
    # vec: (1, ts, 2)
    vec = tf.expand_dims(vec, axis=0)
    # vec: (b*c, ts, 2)
    vector = tf.tile(vec, (b*c, 1, 1))

    # add offset to the input vector
    off_vec = tf.cast(vector, 'float32') + tf.cast(offset, 'float32')

    # safety check
    off_vec = tf.reshape(off_vec, [b*c, ts, 2])
    assert(off_vec.shape[2] == 2)

    # get leftmost and rightmost points
    lp = tf.cast(tf.floor(off_vec), 'int32')
    rp = tf.cast(tf.ceil(off_vec), 'int32')

    lp = warp_map(x, lp)
    rp = warp_map(x, rp)

    weights = off_vec - tf.cast(lp, 'float32')
    l = lp*weights[..., 0] + lp
    r = rp*weights[..., 0] + rp
    v = (r-l) * weights[..., 1] + lp

    return v

def warp_map(x, vec):
    # x: (b, ts)
    # vec: (b, ts, 2)
    b = tf.shape(x)[0]
    ts = tf.shape(x)[1]

    x = tf.reshape(x, (b, ts))
    tpos = vec[..., 0] + vec[..., 1]
    tpos = tf.reshape(tpos, [-1])

    xpos = tf.reshape(x, [-1])
    pdb.set_trace()

    tpos = tf.cast(tpos, 'float32')
    indices = tf.stack([xpos, tpos], axis=-1)
    indices = tf.cast(indices, 'int32')
    output = tf.gather_nd(x, indices)
    output = tf.reshape(output, (b, -1))

    return output
