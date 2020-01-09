"""
A basic nerual network on CIFAR100 dataset using Keras
"""
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.datasets import fashion_mnist
from keras import backend as K

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print(X_train.shape)    #(60000,28,28)
print(y_train.shape)

#defining parameters
batch_size = 32
epochs=50
num_classes= 10
input_shape=(28,28,1)

if K.image_data_format() == "channels_first":
	X_train = X_train.reshape((60000, 784))
	X_test = X_test.reshape((60000, 784))
 
# otherwise, we are using "channels last" ordering, so the design
# matrix shape should be: num_samples x rows x columns x depth
else:
	X_train = X_train.reshape((X_train.shape[0], 784))
	X_test = X_test.reshape((X_test.shape[0], 784))

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#scale data to range [0,1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

print(X_train.shape, X_test.shape)

#Creating the model
model = Sequential()
model.add(Dense(512,input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.20))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.20))
model.add(Dense(num_classes))
model.add(Activation("softmax")) 

print(model.summary())

#optimizer
optimizer = Adam(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, epochs= epochs, shuffle=True)

#evaluation
scores = model.evaluate(X_test, y_test, batch_size= 64, verbose=1)
print("Test loss: ", scores[0])
print("Test Accuracy: ", scores[1])



