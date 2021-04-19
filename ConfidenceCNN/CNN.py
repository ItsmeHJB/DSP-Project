from __future__ import print_function

import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from random import shuffle

import tensorflow as tf

from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# Takes in image title and returns confidence classification
def label_img(img):
    word_label = img.split('_')[0]
    if word_label == "conf":
        return [1, 0]
    else:
        return [0, 1]


def create_data_set(path):
    data = []
    for image in tqdm(os.listdir(path)):
        label = label_img(image)
        image = Image.open(os.path.join(path, image)).convert("RGB")
        data.append([np.array(image), np.array(label)])

    shuffle(data)
    # np.save(path + "_data.npy", data)
    return data


# # Set up starter vals
# IMAGE_WIDTH = 1920
# IMAGE_HEIGHT = 1080
# N_EPOCH = 10
# learning_rate = 1e-3
# MODEL_NAME = "Test model"
#
# Get starter data
print("Creating dataset")
data = create_data_set(Path("../GazemapGen/training"))
train_images = data[0]
train_labels = data[1]
print(len(train_images))
data = create_data_set(Path("../GazemapGen/test"))
test_images = data[0]
test_labels = data[1]
print(len(test_images))

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(train_labels[i])
plt.show()

# https://www.tensorflow.org/tutorials/images/cnn
#
# print("Creating model")
# # Create model
# img_shape = train_data[0][0].shape
# print(img_shape)
#
# convnet = input_data(img_shape, name='inputs')
#
# convnet = conv_2d(convnet, 64, 3, activation='relu')
# convent = max_pool_2d(convnet, 2)
#
# convnet = conv_2d(convnet, 32, 3, activation='relu')
# convent = max_pool_2d(convnet, 2)
#
# convnet = fully_connected(convnet, 512, activation='relu')
#
# convnet = fully_connected(convnet, 2, activation='sigmoid')
# convnet = regression(convnet, optimizer='adam',
#                      name='targets', learning_rate=learning_rate,
#                      loss='binary_crossentropy', metric='accuracy')
#
# model = tflearn.DNN(convnet, tensorboard_dir='log')

# Get images and labels in separate vars
train_img = np.array([i[0] for i in train_data])
train_lab = np.array([i[1] for i in train_data])

test_img = np.array([i[0] for i in test_data])
test_lab = np.array([i[1] for i in test_data])

# Fit the model
model.fit({'inputs': train_img},
          {'targets': train_lab},
          n_epoch=3,
          validation_set=({'inputs': test_img}, {'targets': test_lab}),
          show_metric=True,
          snapshot_step=10,
          run_id=MODEL_NAME,
          batch_size=10)
model.save(MODEL_NAME)

fig = plt.figure(figsize=(10, 6))
import matplotlib.image as mpimg

for num, image in enumerate(test_data):
    # image = mpimg.imread(image)
    y_plot = fig.add_subplot(3, 6, num + 1)
    model_out = model.predict([image[0]])[0][0]
    # print(model_out)
    if model_out == 1:
        label = "CONFIDENT"
    else:
        label = "NON-CONFIDENT"
    y_plot.imshow(image[0])
    plt.title(label)
    y_plot.axis('off')
plt.show()

