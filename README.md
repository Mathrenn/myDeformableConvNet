# myDeformableConvNet

1D implementation of deformable convnets

## What's different

In the paper, the deformable convolution operation is about computing the (bi)linear interpolation between every **input** location and every offset **output** location.

In this version, we augment the input with offsets by applying the convolution operation exclusively to the **input** sequence (and its locations).

On the IMDB sentiment analysis dataset, results are roughly equal (without changing the network architecture) to a standard Conv1D.
