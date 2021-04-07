import tensorflow as tf

from tensorflow.keras import preprocessing, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

import matplotlib.pyplot as plt


# Set up CNN base using a stack of ConvSD and MaxPooling2D layers
model = models.Sequential()
# Convolution
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# Pooling
model.add(layers.MaxPooling2D((2, 2)))
# 2nd layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Display archtecture - output is a 3D tensor of shape
model.summary()

# Adding dense layers on top - these do the classification
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

# Compile the model ready for training
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_images = train_datagen.flow_from_directory(Path("../GazemapGen/training"),
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_images = test_datagen.flow_from_directory(Path("../GazemapGen/test"),
                                               target_size=(64,64),
                                               batch_size=32,
                                               class_mode='binary')

model.fit_generator(train_images,
                    nb_epoch=10,
                    validation_data=test_images,)


# # Evaluate the model
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
#
# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#
# print(test_acc)
