import cv2
import glob
import imgaug.augmenters as iaa
import numpy as np
import os
import re

# Natural sorting function for image filenames
def natural_sort_key(file_path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', file_path)]

def data_augmentation(input_dir, output_dir):

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect and sort file paths in natural order
    all_files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")), key=natural_sort_key)

    # Load images in grayscale
    images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in all_files]
    # Convert images to a format suitable for imgaug
    images = np.array(images)

    # Define an augmentation sequence
    seq = iaa.Sequential([
        iaa.Fliplr(0.0),  # Flip images horizontally with 50% probability
        iaa.Affine(
            rotate=(0, 0),  # Rotate images
            scale=(1.0, 1.2)  # Scale images to 100% to 120%
        ),
        iaa.AdditiveGaussianNoise(scale=(0, 0.01 * 255)),  # Add Gaussian noise
        iaa.LinearContrast((0.8, 1.2)),  # Adjust contrast
        iaa.Crop(percent=(0, 0.1)),  # Randomly crop images
    ])

    # Apply augmentations to each image
    augmented_images = seq(images=images)

    # Save augmented images with original filenames
    for file, img in zip(all_files, augmented_images):
        img = np.clip(img, 0, 255).astype(np.uint8)  # Ensure the image is in the correct format
        output_path = os.path.join(output_dir, f"{os.path.basename(file)}")
        cv2.imwrite(output_path, img)

    print("Augmentation completed and saved to:", output_dir)