import tensorflow as tf
from defConvNN import DeformableConv1D

convlayer = tf.keras.layers.Conv1D(10, 3, input_shape=(None, 5, 3))

x = tf.random.uniform([4,5,3], maxval=10)

print(f"input:\n{x}")
print(f"Convolutional layer:\n{convlayer(x)}\n")
print(f"Deformable convolutional layer:\n{DeformableConv1D(10, 3, offset=convlayer(x), batch_size=4, input_shape=(None, 5, 3))(x)}\n")
