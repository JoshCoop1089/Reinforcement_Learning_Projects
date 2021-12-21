# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 18:47:45 2021

@author: joshc
"""

import tensorflow as tf

# MNIST TF Tutorial Code
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(type(x_train))
print(x_train.shape)
# print(type(x_train))
# model = tf.keras.models.Sequential( [
#     tf.keras.layers.Flatten(input_shape = (28,28)),
#     tf.keras.layers.Dense(128, activation = 'relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10)    
#     ])

# predictions = model(x_train[:1]).numpy()
# print(predictions)
# print(tf.nn.softmax(predictions).numpy())

# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# print(loss_fn(y_train[:1], predictions).numpy())

# model.compile(optimizer='adam',
#               loss = loss_fn,
#               metrics = ['accuracy'])
# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test, verbose=2)

# # Fashion MNIST Tutorial Code
# import numpy as np
# import matplotlib.pyplot as plt

# fashion_mnist = tf.keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print(train_images.shape)
# print(len(train_labels))
# print(train_labels)
# print(type(train_labels))
# print(test_images.shape)
# print(len(test_labels))

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# train_images = train_images / 255.0
# test_images = test_images / 255.0

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10)
# ])

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs=10)
# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# print('\nTest accuracy:', test_acc)

# probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# predictions = probability_model.predict(test_images)
# print(predictions[0])
# print(np.argmax(predictions[0]))
# print(test_labels[0])
