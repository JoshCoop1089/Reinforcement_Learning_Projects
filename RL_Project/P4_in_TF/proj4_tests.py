# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 15:47:16 2021

@author: joshc
"""

import proj4_redux as p4

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


image_name = 'tigerface.jpg'
clusters = 5

img = p4.get_image(image_name)
print( img.shape[0], img.shape[1])
kmeans, labels = p4.get_kmeans_reductions(image_name, clusters)

img_reduced = p4.recolor_image_with_kmeans(kmeans.cluster_centers_, labels, img.shape[0], img.shape[1])
img_gray= p4.convert_grayscale(img)

train_data, train_labels = p4.generate_data(img_gray, labels, clusters, train = True)
print(train_labels[0])

# Take training data and feed it to single layer network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(clusters)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=10)


probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])


test_data, test_labels = p4.generate_data(img_gray, labels, clusters, train = False)
test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

predictions = probability_model.predict(test_data)



def turn_cluster_to_color(cluster_list, current_info):
    # print(current_info)
    cluster_id = current_info.index(1)
    return cluster_list[cluster_id]


image = np.zeros((img.shape[0], img.shape[1], 3))
labels_idx, labels_idx_pred = 0, 0
for i in range(1, img.shape[0]-1):
    for j in range(1, img.shape[1]//2-1):
        image[i][j] = turn_cluster_to_color(kmeans.cluster_centers_, train_labels[labels_idx])
        labels_idx += 1
    for k in range(img.shape[1]//2, img.shape[1]-1):
        image[i][k] = kmeans.cluster_centers_[np.argmax(predictions[labels_idx_pred])]
        labels_idx_pred += 1
right_img_pred = image

# Plot Figures
plt.rcParams['font.size'] = '16'
fig = plt.figure(figsize=(15,15), constrained_layout = True)
preds = fig.add_gridspec(2, 2, hspace=0, wspace=0)
(axs1, axs2), (axs3, axs4) = preds.subplots(sharex='col', sharey='row')
axs1.imshow(img)
axs2.imshow(img_gray)
axs3.imshow(img_reduced)
axs4.imshow(right_img_pred)
plt.suptitle(f"Clockwise from top left: Original, Greyscale, Tensorflow (Predictions on Right Half), K-Means\n Clusters = {clusters}")
plt.show()
plt.close()


