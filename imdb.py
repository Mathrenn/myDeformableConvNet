from __future__ import print_function

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.datasets import imdb
from tensorflow.python.keras.engine.base_layer import InputSpec

# set parameters:
max_features = 10000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
I = Input(shape=(maxlen,))
#I = Input(batch_shape=(batch_size, maxlen))
x = Embedding(max_features, embedding_dims)(I)
x = Dropout(0.2)(x)

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
x = DeformableConv1D(filters, kernel_size)(x)
x = Activation('relu')(x)

# we use max pooling:
x = GlobalMaxPooling1D()(x)

# We add a vanilla hidden layer:
x = Dense(hidden_dims)(x)
x = Dropout(0.2)(x)
x = Activation('relu')(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
x = Dense(1)(x)
x = Activation('sigmoid')(x)

model = Model(inputs=I, outputs=x)

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

