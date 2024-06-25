import os
import cv2
import numpy as np
import time

def crop_and_resize_images(input_folder, output_folder, target_size=(400, 400)):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    image_files = os.listdir(input_folder)

    for file in image_files:
        # Read the image using OpenCV
        image_path = os.path.join(input_folder, file)
        image = cv2.imread(image_path)
        
        
        # Resize the image to the target size
        resized_image = cv2.resize(image, target_size)

        # Perform cropping (adjust these values according to your needs)
        cropped_image = resized_image[50:350, 50:350]

        # Resize the image to the target size
        # resized_image2 = cv2.resize(cropped_image, (224,224))

        # Save the resized image to the output folder
        output_path = os.path.join(output_folder, file)
        cv2.imwrite(output_path, cropped_image)

        print(f"Processed: {file}")

# Usage example
input_folder  = "your input folder"

output_folder = "CropedResizedImages"

start_time = time.time()

crop_and_resize_images(input_folder, output_folder)

#Augmentation
# Define the input and output directories
input_dir = output_folder
output_dir = "Augmented_Images"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the augmentation transformations
def augment_data(img):
    # Randomly flip the image horizontally
    if np.random.randint(2) == 0:
        img = cv2.flip(img, 1)

    # Randomly flip the image vertically
    if np.random.randint(2) == 0:
        img = cv2.flip(img, 0)

    # Randomly rotate the image
    angle = np.random.randint(-20, 20)
    scale = 1.0
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
    img = cv2.warpAffine(img, M, (cols, rows))

    # Randomly shift the image horizontally and vertically
    tx = np.random.randint(-20, 20)
    ty = np.random.randint(-20, 20)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M, (cols, rows))

    return img

# Loop through all the files in the input directory
for filename in os.listdir(input_dir):
    # Load the image using OpenCV
    img = cv2.imread(os.path.join(input_dir, filename))

    # Apply data augmentation to the image
    augmented_images = [augment_data(img) for _ in range(5)]

    # Save the new augmented images in the output directory
    for i, augmented_img in enumerate(augmented_images):
        save_filename = os.path.splitext(filename)[0] + f'_{i}.png'
        save_path = os.path.join(output_dir, save_filename)
        cv2.imwrite(save_path, augmented_img)

end_time = time.time()

elapsed_time = end_time - start_time
print("Preprocessing Elapsed time:", elapsed_time, "seconds")