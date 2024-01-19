import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np

# Load the trained model
model = load_model('best_model.h5')  # Update with the path to your trained model

# Load the sample submission file
sample_submission = pd.read_csv('C:/Users/vishv/OneDrive/Documents/extracting-attributes-from-fashion-images-jan-2024/sample_submission.csv')

# Extract the file names from the sample submission
test_file_names = sample_submission['file_name']

# Preprocess the test data
test_images = []

for file_name in test_file_names:
    img_path = 'test/' + file_name  # Update with the path to your test images
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Rescale pixel values to the range [0, 1]
    test_images.append(img_array)

test_images = np.array(test_images)

# Make predictions on the test set
predictions = model.predict(test_images)
predicted_labels = label_encoder.inverse_transform(predictions.argmax(axis=1))

# Update the sample submission DataFrame
sample_submission['label'] = predicted_labels

# Save the updated submission DataFrame to a new CSV file
sample_submission.to_csv('submission.csv', index=False)
