import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import time

# Define the input directory and the path to the output CSV file

input_dirs = ["Augmented Images Directory"]
output_file = "VGG-16.csv"

# Load the VGG16 model and remove the final classification layer
base_model = VGG16(weights='imagenet', include_top=False , pooling='max')
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# Define a function to extract features from an image
def extract_features(img_path):
    # Load the image using OpenCV and resize it to (224, 224)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    
    # Apply median filter to the image
    img = cv2.medianBlur(img, 3)

    # Preprocess the image using the VGG16 preprocess_input function
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)

    # Extract features from the image using the VGG16 model
    features = model.predict(x)

    return features.flatten()

### progress
def print_progress(progress):
    print("Processing file", progress["file_count"], "of", progress["total_files"], ":", progress["filename"])

# Loop through all the input directories and files to count the total number of files
total_files = 0
for input_dir in input_dirs:
    for filename in os.listdir(input_dir):
        total_files += 1
###
start_time = time.time()

# Loop through all the input directories
features = []
for input_dir in input_dirs:
    
    ### progress
    progress = {"input_dir": input_dir, "total_files": total_files, "file_count": 0}
    ###
    
    # Loop through all the files in the input directory
    for filename in os.listdir(input_dir):
        
        ### progress
        progress["file_count"] += 1
        progress["filename"] = filename
        print_progress(progress)
        ###
        
        # Extract features from the image and append them to the features list
        img_path = os.path.join(input_dir, filename)
        img_features = extract_features(img_path)
        label = "yes" if "yes" in input_dir else "no"
        features.append(np.concatenate((img_features, [label])))
  
end_time = time.time()

elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")

# Convert the features list to a Pandas DataFrame and save it to a CSV file
df = pd.DataFrame(features)
df.to_csv(output_file, index=False)