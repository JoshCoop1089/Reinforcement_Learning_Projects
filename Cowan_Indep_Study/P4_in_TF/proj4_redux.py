# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 21:00:59 2021

@author: joshc

----------------------
Recreate Proj 4 with TF
    How do i get the data
    how do i organize the data
    what are my training sets/test sets
    how do i measure accuracy
    how do i turn the model into a predictive output
    how do i check my results visually

"""

from sklearn.cluster import KMeans
import matplotlib.image as mpimg
import numpy as np
import os


def get_image(image_name):
    """
    Returns an NxMx3 array
    """
    path = 'Images\\' + image_name 
    absolute_path = os.path.join(os.getcwd(), path)
    img = mpimg.imread(absolute_path)
    return img

def get_kmeans_reductions (image_name, clusters):
# Run K-means to get lower dimensional coloring
    
    img = get_image(image_name)
    img1 = img/255
    
    # Flatten the image into one dimension with RGB values
    img1=img1.reshape((img.shape[1]*img.shape[0],3))
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(img1)
    labels = kmeans.predict(img1)
    return kmeans, labels

def recolor_image_with_kmeans(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


# Elbow Method for IDing Kmeans Clustering
# distortions = []
# for i in range(1, 11):
#     km = KMeans(
#         n_clusters=i, init='random',
#         n_init=10, max_iter=300,
#         tol=1e-04, random_state=0
#     )
#     km.fit(img1)
#     distortions.append(km.inertia_)

# # plot
# plt.imshow(img)
# plt.show()
# plt.plot(range(1, 11), distortions, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Distortion')
# plt.show()

# All 4 Images in test file elbow out at k_means_clusters = 4


def convert_grayscale(img): #only use jpgs
    dimensions = img.shape
    grayscale_image = np.zeros(dimensions, int)
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            grayscale_image[i][j] = 0.21 * img[i][j][0] + 0.72 * img[i][j][1] + 0.07 * img[i][j][2]
    
    return grayscale_image

# Create training data/testing data dictionaries
#(center of patch location) -> 3x3 pixel patch, one hot encoded cluster value for center of patch
def generate_data(image, centers, clusters, train = True):
    im_rows, im_cols = image.shape[0], image.shape[1]
    cents = np.reshape(centers, [im_rows, im_cols])
    data = []
    labels = []
    # print(im_rows, im_cols)
    for i in range(1,im_rows-1):
        start = 1
        end = im_cols//2-1
        if not train:
            start = im_cols//2
            end = im_cols-1
        for j in range(start, end):
           
            patch = [(i-1, j-1), (i-1, j), (i-1, j+1),
                      (i, j-1),   (i,j),    (i, j+1),
                      (i+1, j-1), (i+1, j), (i+1, j+1)]
            patch_greys = []
            p_g_2 = [image[i][j][0]/255.0 for (i,j) in patch]
            patch_greys.extend(p_g_2)
            
            cluster_num = cents[i][j]
            cluster_encode = [0 if x != cluster_num else 1 for x in range(clusters)]
            data.append(patch_greys)
            labels.append(cluster_encode)
    return data, labels
