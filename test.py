import tensorflow as tf
from defConvNN import DeformableConv1D
tf.enable_eager_execution()

convlayer = tf.keras.layers.Conv1D(10, 3, input_shape=(None, 5, 3))
defconvlayer = DeformableConv1D(10, 3, input_shape=(None, 5, 3))

x = tf.random.uniform([4,5,3], maxval=10)

print(f"input:\n{x}")
print(f"Convolutional layer:\n{convlayer(x)}\n")
print(f"Deformable convolutional layer:\n{defconvlayer(x)}\n")
#print(f"Deformable convolutional layer:\n{convlayer(defconvlayer(x))}\n")
