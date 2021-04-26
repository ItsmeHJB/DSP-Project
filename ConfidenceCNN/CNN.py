# Simple CNN to predict confidence in a binary setting

import os
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
                    steps_per_epoch=4,
                    epochs=8,
                    verbose=1,
                    validation_data=validation_generator,
                    validation_steps=8)

# Test the accuracy
model.evaluate(validation_generator)

STEP_SIZE_TEST = validation_generator.n // validation_generator.batch_size
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
