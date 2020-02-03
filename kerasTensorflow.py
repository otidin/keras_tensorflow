import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline
import keras
keras.backend.backend()
# from keras.datasets import fashion_mnist
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
# plt.matshow(x_train[0])
plt.matshow(x_train[0])

imatlax_train=x_train/255
x_test=x_test/255
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

model=Sequential()

model.add(Flatten(input_shape=[28,28]))
model.add(Dense(200, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

# model = Sequential([
#     Dense(32, input_shape=(784,)),
#     Activation('relu'),
#     Dense(10),
#     Activation('softmax'),˚
# ])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(x_test.shape)

yp =model.predict(x_test)

np.argmax(yp[0])

model.evaluate(x_test,y_test)