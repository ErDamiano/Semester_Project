import glob
import os
import cv2
import re

from ChooseWindow import choose_window
from DataAugmentation import data_augmentation
from GaussianNoise import apply_noise_to_directory
'''
# Set initial parameters to crop the images around x y with a window size of Nx and Ny
Nx = 512  # Default width window
Ny = 512  # Default height window
x, y = 1100, 1000  # Default center coordinates for the crop window

# List of directories containing images
# Natural sorting function for image filenames
def natural_sort_key(file_path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', file_path)]

input_dirs = [
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/stabilizedFrames/MN_001",
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/stabilizedFrames/MN_002",
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/stabilizedFrames/MN_003",
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/stabilizedFrames/MN_451",
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/stabilizedFrames/MN_452",
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/stabilizedFrames/MN_453",
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/stabilizedFrames/MN_901",
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/stabilizedFrames/MN_902",
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/stabilizedFrames/MN_903",
]

# Define the output directory for collected images
output_dir = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/To be processed"
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

# Collect and save images into the output directory
image_count = 0
for input_dir in input_dirs:
    # Get image paths and sort them in natural order
    image_paths = sorted(glob.glob(os.path.join(input_dir, '*.jpg')), key=natural_sort_key)
    for image_path in image_paths:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read {image_path}, skipping.")
            continue

        # Save the image to the output directory with a unique filename
        output_path = os.path.join(output_dir, f"image_{image_count}.jpg")
        cv2.imwrite(output_path, image)
        image_count += 1

# Input and output directories to apply cropping
input_dir_window_size = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/To be processed"
output_dir_window_size = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/croppedFrames"
# Choose the window size to be cropped out of the stabilized images
choose_window(input_dir_window_size, output_dir_window_size, x, y, Nx, Ny)

'''
# Call the Data Augmentation function to make the frames more different from each other
# and avoid overfitting
input_dir_augment = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/croppedFrames"
output_dir_augment = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/AugmentedFrames"
data_augmentation(input_dir_augment, output_dir_augment)

# Adds a mask of Gaussian patches to feed to the Algorithm which will try to remove it and restore the image
input_dir_noisy = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/AugmentedFrames"
output_dir_noisy = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/All processed images"
apply_noise_to_directory(input_dir_noisy, output_dir_noisy, 16, 0.4)

