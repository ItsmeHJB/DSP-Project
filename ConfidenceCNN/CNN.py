# from __future__ import print_function
#
# import tflearn
# from tflearn.layers.core import input_data, fully_connected, dropout
# from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.estimator import regression
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from PIL import Image
# from pathlib import Path
# from tqdm import tqdm
# from random import shuffle
#
# import tensorflow as tf
#
# from tensorflow.keras import layers, models
# import matplotlib.pyplot as plt
#
#
# # Takes in image title and returns confidence classification
# def label_img(img):
#     word_label = img.split('_')[0]
#     if word_label == "conf":
#         return [1, 0]
#     else:
#         return [0, 1]
#
#
# def create_data_set(path):
#     data = []
#     for image in tqdm(os.listdir(path)):
#         label = label_img(image)
#         image = Image.open(os.path.join(path, image)).convert("RGB")
#         data.append([np.array(image), np.array(label)])
#
#     shuffle(data)
#     # np.save(path + "_data.npy", data)
#     return data
#
#
# # # Set up starter vals
# IMAGE_WIDTH = 1920
# IMAGE_HEIGHT = 1080
# N_EPOCH = 10
# learning_rate = 1e-3
# MODEL_NAME = "Test model"
# #
# # Get starter data
# print("Creating dataset")
# train_data = create_data_set(Path("../GazemapGen/training"))
# test_data = create_data_set(Path("../GazemapGen/test"))
# # data = create_data_set(Path("../GazemapGen/training"))
# # train_images = data[0]
# # train_labels = data[1]
# # print(len(train_images))
# # data = create_data_set(Path("../GazemapGen/test"))
# # test_images = data[0]
# # test_labels = data[1]
# # print(len(test_images))
#
# # plt.figure(figsize=(10,10))
# # for i in range(25):
# #     plt.subplot(5,5,i+1)
# #     plt.xticks([])
# #     plt.yticks([])
# #     plt.grid(False)
# #     plt.imshow(train_images[i], cmap=plt.cm.binary)
# #     # The CIFAR labels happen to be arrays,
# #     # which is why you need the extra index
# #     plt.xlabel(train_labels[i])
# # plt.show()
#
# # https://www.tensorflow.org/tutorials/images/cnn
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
#
# # Get images and labels in separate vars
# train_img = np.array([i[0] for i in train_data])
# train_lab = np.array([i[1] for i in train_data])
#
# test_img = np.array([i[0] for i in test_data])
# test_lab = np.array([i[1] for i in test_data])
#
# # Fit the model
# model.fit({'inputs': train_img},
#           {'targets': train_lab},
#           n_epoch=3,
#           validation_set=({'inputs': test_img}, {'targets': test_lab}),
#           show_metric=True,
#           snapshot_step=10,
#           run_id=MODEL_NAME,
#           batch_size=10)
# model.save(MODEL_NAME)
#
# fig = plt.figure(figsize=(10, 6))
# import matplotlib.image as mpimg
#
# for num, image in enumerate(test_data):
#     # image = mpimg.imread(image)
#     y_plot = fig.add_subplot(3, 6, num + 1)
#     model_out = model.predict([image[0]])[0][0]
#     # print(model_out)
#     if model_out == 1:
#         label = "CONFIDENT"
#     else:
#         label = "NON-CONFIDENT"
#     y_plot.imshow(image[0])
#     plt.title(label)
#     y_plot.axis('off')
# plt.show()

import os
from pathlib import Path

# Get training images from dirs
train_conf_dir = os.path.join(Path("../GazemapGen/training/conf"))
train_non_conf_dir = os.path.join(Path("../GazemapGen/training/non_conf"))

# Get test images from dirs
test_conf_dir = os.path.join(Path("../GazemapGen/test/conf"))
test_non_conf_dir = os.path.join(Path("../GazemapGen/test/non_conf"))

# Print total images
print('total training confident images:', len(os.listdir(train_conf_dir)))
print('total training non-confident images:', len(os.listdir(train_non_conf_dir)))
print('total validation confident images:', len(os.listdir(test_conf_dir)))
print('total validation non-confident images:', len(os.listdir(test_non_conf_dir)))

# Data pre-processing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images rescaled to 1/225
train_datagen = ImageDataGenerator(rescale=1 / 255)
validation_datagen = ImageDataGenerator(rescale=1 / 255)

# Flow training images in batches of 20
train_generator = train_datagen.flow_from_directory(
    Path("../GazemapGen/training"),  # This is the source directory for training images
    classes=['conf', 'non_conf'],
    target_size=(200, 200),  # All images will be resized to 200x200
    batch_size=20,
    # Use binary labels
    class_mode='binary')

# Flow validation images in batches of 10
validation_generator = validation_datagen.flow_from_directory(
    Path("../GazemapGen/test"),  # This is the source directory for training images
    classes=['conf', 'non_conf'],
    target_size=(200, 200),  # All images will be resized to 200x200
    batch_size=10,
    # Use binary labels
    class_mode='binary',
    shuffle=False)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
import numpy as np
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
# from scipy import interp
from sklearn.metrics import roc_auc_score

# Build model using flattening and dense layers, 1 in non_conf, 0 is conf here
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(200, 200, 3)),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])
model.summary()  # Print model shape
# Compile the model together
model.compile(optimizer=tf.optimizers.Adam(),  # Adam outmates the learning-rate tuning for us. He's a very kind person.
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Fit the model to the images we have
history = model.fit(train_generator,
                    steps_per_epoch=8,
                    epochs=15,
                    verbose=1,
                    validation_data=validation_generator,
                    validation_steps=8)

# Test the accuracy
model.evaluate(validation_generator)

STEP_SIZE_TEST=validation_generator.n//validation_generator.batch_size
validation_generator.reset()
preds = model.predict(validation_generator,
                      verbose=1)

fpr, tpr, _ = roc_curve(validation_generator.classes, preds)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
