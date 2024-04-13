import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin")
os.add_dll_directory("C:/Users/James/Documents/cuda/bin")

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def preprocess_image(path):
    img = load_img(path, target_size = (img_height, img_width))
    a = img_to_array(img)
    a = np.expand_dims(a, axis = 0)
    a /= 255.
    return a

# Standardise image width and height
img_width = 256
img_height = 256

batch_size = 16

TRAINING_DIR = "C:/Users/James/Desktop/mush2024/constellation_data/train"

# Rotate, rescale and zoom to create a wider range of images from different angle
train_datagen = ImageDataGenerator(rescale=1/255.0, rotation_range=30, zoom_range=0.4)


train_generator = train_datagen.flow_from_directory(TRAINING_DIR, batch_size=batch_size, class_mode='categorical', target_size=(img_height, img_width))

# Same for validation
VALIDATION_DIR = "C:/Users/James/Desktop/mush2024/constellation_data/validation"

validation_datagen = ImageDataGenerator(rescale=1/255.0, rotation_range=30, zoom_range=0.4)

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, batch_size=batch_size, class_mode='categorical', target_size=(img_height, img_width))


callbacks = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
best_model_file = 'C:/Users/James/Desktop/mush2024/CNN_best_weights.h5'
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose=1, save_best_onoly=True)

# Instantiate the model
# TODO: Test this and should produce expected outcome
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)), MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'), MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'), MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'), MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'), MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dense(16, activation='softmax')
])

model.summary()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[ 'accuracy'])

# Epochs should be increased to fully train (but will take longer)
history = model.fit_generator(train_generator, epochs=20, verbose=1, validation_data=validation_generator, callbacks=[best_model])

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))

fig = plt.figure(figsize=(14,7))
plt.plot(epochs, acc, 'r', label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')
plt.show()

fig2 = plt.figure(figsize=(14,7))
plt.plot(epochs, loss, 'r', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and validation loss')





