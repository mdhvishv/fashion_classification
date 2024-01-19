'''import os
import cv2
import pandas as pd
import numpy as np

# Load the original training data and labels
train_df = pd.read_csv("C:/Users/vishv/OneDrive/Documents/extracting-attributes-from-fashion-images-jan-2024/train.csv")

# Create a new directory to store cropped trouser images
output_cropped_dir = "train_cropped_trousers"
os.makedirs(output_cropped_dir, exist_ok=True)


# Function to crop trouser images
def crop_trouser_image(image_path, output_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply image processing to detect and extract the trouser region (customize as needed)
    # Example: Using a simple vertical crop assuming the trouser is typically in the lower part of the image
    height, width = img.shape[:2]
    cropped_img = img[height // 2:, :]

    # Save the cropped image
    cv2.imwrite(output_path, cropped_img)


# Iterate through the original images and crop to focus on the trouser
for index, row in train_df.iterrows():
    file_name = row['file_name']
    original_path = f"train/{file_name}"
    output_path = os.path.join(output_cropped_dir, file_name)

    crop_trouser_image(original_path, output_path)

# Update the train_df to contain entries corresponding to the cropped trouser images
train_cropped_df = pd.DataFrame({
    'file_name': os.listdir(output_cropped_dir),
    'label': train_df['label'].values  # Assuming the labels remain the same
})

train_cropped_df.to_csv("train_cropped_trousers.csv", index=False)

# Now, train your model using the "train_cropped_trousers.csv" file and the "train_cropped_trousers" directory.
# Adjust the code accordingly to load the new dataset and perform training.
import os
import shutil
import pandas as pd

# Load the original training data and labels
train_df = pd.read_csv("C:/Users/vishv/OneDrive/Documents/extracting-attributes-from-fashion-images-jan-2024/train.csv")

# Create a new directory to store only trouser images
output_trouser_dir = "train_trousers"
os.makedirs(output_trouser_dir, exist_ok=True)

# Iterate through the original images and copy only the trousers to the new directory
for index, row in train_df.iterrows():
    file_name = row['file_name']
    label = row['label']

    # Assuming label 0 corresponds to the trouser class
    if label == 0:
        original_path = f"train/{file_name}"
        new_path = os.path.join(output_trouser_dir, file_name)
        shutil.copyfile(original_path, new_path)

# Update the train_df to contain only the trouser images
train_trouser_df = train_df[train_df['label'] == 0]
train_trouser_df.to_csv("train_trousers.csv", index=False)

# Now, train your model using the "train_trousers.csv" file and the "train_trousers" directory.
# Adjust the code accordingly to load the new dataset and perform training.
'''
import shutil
import os
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Load the original training data and labels
train_df = pd.read_csv("C:/Users/vishv/OneDrive/Documents/extracting-attributes-from-fashion-images-jan-2024/train.csv")

# Create folders for complete outfits and just trousers
output_complete_outfit_dir = "train_complete_outfit"
output_trousers_only_dir = "train_trousers_only"
os.makedirs(output_complete_outfit_dir, exist_ok=True)
os.makedirs(output_trousers_only_dir, exist_ok=True)

# Load a pre-trained ResNet50 model (you can choose a different model based on your requirements)
model = ResNet50(weights='imagenet')

# Function to classify and save images based on the pre-trained model's predictions
def classify_and_save_image(image_path, output_complete_dir, output_trousers_dir):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    predicted_label = decoded_predictions[0][1]

    # Assuming a simple condition: if the predicted label contains "trouser", save to trousers folder
    if "trouser" in predicted_label.lower():
        output_path = os.path.join(output_trousers_dir, file_name)
    else:
        output_path = os.path.join(output_complete_dir, file_name)

    # Copy the original image to the corresponding folder
    shutil.copyfile(image_path, output_path)

# Iterate through the original images and classify them
for index, row in train_df.iterrows():
    file_name = row['file_name']
    original_path = f"train/{file_name}"

    classify_and_save_image(original_path, output_complete_outfit_dir, output_trousers_only_dir)
