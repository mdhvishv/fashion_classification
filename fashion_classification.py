'''import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Load the training data and labels
train_df = pd.read_csv("C:/Users/vishv/OneDrive/Documents/extracting-attributes-from-fashion-images-jan-2024/train.csv")
train_images = []

for file_name in train_df['file_name']:
    img_path = f"train/{file_name}"
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    train_images.append(img_array)

train_images = np.array(train_images)
train_labels = np.array(train_df['label'])

# Split the training data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

# Data augmentation for training set
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)

datagen.fit(train_images)

# Build the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))  # 7 classes for trouser fit types

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define a ModelCheckpoint callback to save the best model during training
checkpoint_callback = ModelCheckpoint("best_model.h5", save_best_only=True)

# Train the model with data augmentation
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=32),
    epochs=50,
    validation_data=(val_images, val_labels),
    callbacks=[checkpoint_callback]
)

# Load the best model for evaluation
model.load_weights("best_model.h5")

# Evaluate the model on the test set
test_df = pd.read_csv("C:/Users/vishv/OneDrive/Documents/extracting-attributes-from-fashion-images-jan-2024/sample_submission.csv")
test_images = []

for file_name in test_df['file_name']:
    img_path = f"test/{file_name}"
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    test_images.append(img_array)

test_images = np.array(test_images)

# Make predictions
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate and print the accuracy on the test set
test_accuracy = np.sum(predicted_labels == test_df['label']) / len(test_df)
print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")

# Create a DataFrame for submission
submission_df = pd.DataFrame({
    'file_name': test_df['file_name'],
    'label': predicted_labels
})

# Save the submission DataFrame to a CSV file
submission_df.to_csv("submission.csv", index=False)

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Load the training data and labels
train_df = pd.read_csv("C:/Users/vishv/OneDrive/Documents/extracting-attributes-from-fashion-images-jan-2024/train.csv")
train_images = []

for file_name in train_df['file_name']:
    img_path = f"train/{file_name}"
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    train_images.append(img_array)

train_images = np.array(train_images)
train_labels = np.array(train_df['label'])

# Split the training data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

# Data augmentation for training set
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False
)

datagen.fit(train_images)

# Build the CNN model with increased complexity
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(7, activation='softmax'))  # 7 classes for trouser fit types

# Compile the model with adjusted learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define a ModelCheckpoint callback to save the best model during training
checkpoint_callback = ModelCheckpoint("best_model.h5", save_best_only=True)

# Train the model with data augmentation and increased epochs
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=32),
    epochs=30,
    validation_data=(val_images, val_labels),
    callbacks=[checkpoint_callback]
)

# Load the best model for evaluation
model.load_weights("best_model.h5")

# Evaluate the model on the test set
test_df = pd.read_csv("C:/Users/vishv/OneDrive/Documents/extracting-attributes-from-fashion-images-jan-2024/sample_submission.csv")
test_images = []

for file_name in test_df['file_name']:
    img_path = f"test/{file_name}"
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    test_images.append(img_array)

test_images = np.array(test_images)

# Make predictions
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate and print the accuracy on the test set
test_accuracy = np.sum(predicted_labels == test_df['label']) / len(test_df)
print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")

# Create a DataFrame for submission
submission_df = pd.DataFrame({
    'file_name': test_df['file_name'],
    'label': predicted_labels
})

# Save the submission DataFrame to a CSV file
submission_df.to_csv("submission.csv", index=False)
'''
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.utils import class_weight

# Load the training data and labels
train_df = pd.read_csv("C:/Users/vishv/OneDrive/Documents/extracting-attributes-from-fashion-images-jan-2024/train.csv")
train_images = []

for file_name in train_df['file_name']:
    img_path = f"train/{file_name}"
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    train_images.append(img_array)

train_images = np.array(train_images)
train_labels = np.array(train_df['label'])

# Split the training data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

# Data augmentation for the training set
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False
)

datagen.fit(train_images)

# Build the model using EfficientNetB0 as the base model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
base_model.trainable = False

# Create your custom classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax')  # Assuming 7 classes for trouser fit types
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Calculate class weights to handle class imbalance
class_weights = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)

# Define a ModelCheckpoint callback to save the best model during training
checkpoint_callback = ModelCheckpoint("best_model2.h5", save_best_only=True)

# Train the model with data augmentation and class weights
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=32),
    epochs=30,
    validation_data=(val_images, val_labels),
    callbacks=[checkpoint_callback],
    class_weight=dict(enumerate(class_weights))
)

# Load the best model for evaluation
model.load_weights("best_model2.h5")

test_df = pd.read_csv("C:/Users/vishv/OneDrive/Documents/extracting-attributes-from-fashion-images-jan-2024/sample_submission.csv")
test_images = []

# Load and preprocess test data
for file_name in test_df['file_name']:
    img_path = f"test/{file_name}"
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    test_images.append(img_array)

test_images = np.array(test_images)

# Make predictions
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate and print the accuracy on the test set
test_accuracy = np.sum(predicted_labels == test_df['label']) / len(test_df)
print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")

# Create a DataFrame for submission
submission_df = pd.DataFrame({
    'file_name': test_df['file_name'],
    'label': predicted_labels
})

# Save the submission DataFrame to a CSV file
submission_df.to_csv("submission1.csv", index=False)
