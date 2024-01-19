import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

img_height, img_width = 224, 224
batch_size = 32
epochs = 30
data_directory = "train"
test_data_directory = "test"

# Load the training data CSV
train_df = pd.read_csv('C:/Users/vishv/OneDrive/Documents/extracting-attributes-from-fashion-images-jan-2024/train.csv')

# Assuming you have seven classes (0 to 6)
num_classes = 7

# Assuming you have a DataFrame with test file names
test_file_names_df = pd.read_csv('C:/Users/vishv/OneDrive/Documents/extracting-attributes-from-fashion-images-jan-2024/sample_submission.csv')

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    preprocessing_function=preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=data_directory,
    x_col='file_name',
    y_col='label',
    class_mode='raw',
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

# Using ResNet50 as the base model
pretrained_model_for_demo = ResNet50(
    include_top=False,
    input_shape=(img_height, img_width, 3),
    weights='imagenet',
    pooling='avg'
)

demo_model = Sequential([
    pretrained_model_for_demo,
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])

for layer in pretrained_model_for_demo.layers:
    layer.trainable = False

demo_model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Add callbacks for better training control
checkpoint = ModelCheckpoint('best_model1.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001, verbose=1)

# Train the model
history = demo_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Load the best model weights
demo_model.load_weights('best_model1.h5')

# Make predictions on the test set
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=preprocess_input
)

test_generator = test_datagen.flow_from_dataframe(
    test_file_names_df,
    directory=test_data_directory,
    x_col='file_name',
    y_col=None,
    class_mode=None,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

predictions = demo_model.predict(test_generator)
predicted_labels = np.argmax(predictions, axis=1)

# Create a DataFrame for submission
submission_df = pd.DataFrame({
    'file_name': test_file_names_df['file_name'],
    'label': predicted_labels
})

# Ensure class labels start from 0
submission_df['label'] = submission_df['label'] - 1

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission_resnet501.csv', index=False)
