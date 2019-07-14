import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
import pdb

max_features = 10000
maxlen = 500

print("Downloading dataset...")

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print("Dataset downloaded.")

print("Pad sequences (sample x time)")
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

from keras.layers import Dense, Input, Embedding 
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D
from keras import models
from keras.optimizers import RMSprop
from defConvNN import DeformableConv1D

I = Input(shape=(maxlen,))
x = Embedding(max_features, 128)(I)
x = Conv1D(32, 7, activation='relu')(x)
# input shape error
x = DeformableConv1D(32, activation='relu')(x)
x = MaxPooling1D(5)(x)
#x = Conv1D(32, 7, activation='relu')(x)
#x = DeformableConv1D(32, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(1)(x)

model = models.Model(inputs=I, outputs=x)

model.summary()

model.compile(optimizer=RMSprop(lr=1e-4),
                loss='binary_crossentropy',
                metrics=['acc'])

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
