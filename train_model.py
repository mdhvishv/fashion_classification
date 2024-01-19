'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, optimizers

# Load and preprocess training data
df_train = pd.read_csv("C:/Users/vishv/OneDrive/Documents/extracting-attributes-from-fashion-images-jan-2024/train.csv")
label_encoder = LabelEncoder()
df_train['label'] = label_encoder.fit_transform(df_train['label'])

train_data, val_data = train_test_split(df_train, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

# Load pre-trained EfficientNetB0 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Build a custom model on top of the pre-trained base
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')  # Assuming 7 classes
])

model.compile(optimizer=optimizers.Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit_generator(
    datagen.flow_from_dataframe(train_data, directory='train', x_col='file_name', y_col='label', target_size=(224, 224), batch_size=32, class_mode='raw'),
    epochs=10,
    validation_data=datagen.flow_from_dataframe(val_data, directory='train', x_col='file_name', y_col='label', target_size=(224, 224), batch_size=32, class_mode='raw')
)

# Save the trained model
model.save('efficientnet_model.h5')'''

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

img_height, img_width = 224, 224
batch_size = 32
epochs = 10
data_directory = "train"
test_data_directory = "test"

# Load the training data CSV
train_df = pd.read_csv('C:/Users/vishv/OneDrive/Documents/extracting-attributes-from-fashion-images-jan-2024/train.csv')

# Assuming you have seven classes (0 to 6)
num_classes = 7

# Assuming you have a DataFrame with test file names
test_file_names_df = pd.read_csv('C:/Users/vishv/OneDrive/Documents/extracting-attributes-from-fashion-images-jan-2024/sample_submission.csv')

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=data_directory,
    x_col='file_name',
    y_col='label',
    class_mode='raw',  # 'sparse' for integer labels
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='training'
)

validation_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=data_directory,
    x_col='file_name',
    y_col='label',
    class_mode='raw',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='validation'
)

# Using EfficientNetB0 as the base model
pretrained_model_for_demo = tf.keras.applications.EfficientNetB0(
    include_top=False,
    input_shape=(img_height, img_width, 3),
    weights='imagenet',
    pooling='avg'
)

demo_model = Sequential([
    pretrained_model_for_demo,
    Flatten(),
    Dropout(0.2),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Freeze the layers of the base model
for layer in pretrained_model_for_demo.layers:
    layer.trainable = False

demo_model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = demo_model.fit(train_generator, validation_data=validation_generator, epochs=epochs)

# Make predictions on the test set
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    test_file_names_df,
    directory=test_data_directory,
    x_col='file_name',
    y_col=None,
    class_mode=None,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False  # Important: Keep the order for submission
)

predictions = demo_model.predict(test_generator)
predicted_labels = np.argmax(predictions, axis=1)

# Create a DataFrame for submission
submission_df = pd.DataFrame({
    'file_name': test_file_names_df['file_name'],
    'label': predicted_labels
})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission1.csv', index=False)
