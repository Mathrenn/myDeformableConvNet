# Following https://keras.io/layers/writing-your-own-keras-layers/

from keras import backend as K
from keras.layers import Layer, Conv1D
from scipy import interpolate
import numpy as np

# Deformable 1D Convolution
class DeformableConv1D(Conv1D):

    def __init__(self, filters, **kwargs):
        super(DeformableConv1D, self).__init__(filters=filters*2, kernel_size=3,
                                               dilation=1, padding='same',
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
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

"""
    interpolation kernel
    parameters:
        q is a location in the input feature map
        p is a fractional location (p = p0 + pn + dpn)
"""
def G(q, p):
    return max(0, 1 - abs(q - p))

"""
    interpolation operation
    parameters:
        x is the input feature map
        p is a fractional location (p = p0 + pn + dpn)
    G is non-zero only for a few qs, so it is fast to compute
"""
def X(x, p):
    Xsum = 0
    for q, v in enumerate(x):
        Xsum = Xsum + G(q, p) * x[q]
    return Xsum

"""
    linear interpolation
    parameters:
        x is the input feature vector
        offset is the offset vector
"""
def linearInterpolation(x, offset):
    y = np.zeros(x.shape())
    offset = offset[2]
    for p0, v0 in enumerate(y):
        for pn, vn in enumerate(y):
            v0 = v0 + w[pn] * X(x, p0 + pn + offset[pn])
    return y