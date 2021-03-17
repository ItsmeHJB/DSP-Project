# Script used to save CIFAR-10 dataset images in the same form as used in the ActiVAte

from tensorflow.keras.datasets import cifar10
from PIL import Image as im

# load the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print shape of each variable
print("train image shape:", x_train.shape)  # output (50000, 32, 32, 3)
print("train label shape:", y_train.shape)  # output (50000, 1)
print("test image shape:", x_test.shape)  # output (10000, 32, 32, 3)
print("test label shape:", y_test.shape)  # output (10000, 1)

for index, img in enumerate(x_train, start=0):
    data = im.fromarray(img)
    data.save('static/images/cifar10_keras/train/CIFAR10_image_(' + str(index) + ').jpg')

for index, img in enumerate(x_test, start=0):
    data = im.fromarray(img)
    data.save('static/images/cifar10_keras/test/CIFAR10_image_(' + str(index) + ').jpg')

