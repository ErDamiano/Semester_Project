import glob
import os
import cv2
import re

from ChooseWindow import choose_window
from DataAugmentation import data_augmentation
from GaussianNoise import apply_noise_to_directory
from LoadInputAndTarget import load_images
from AutoEncoder import create_autoencoder
# from EvaluateEncoder import evaluate_autoencoder
from tensorflow.keras.models import load_model

# Set initial parameters to crop the images around x y with a window size of Nx and Ny
Nx = 32  # Default width window
Ny = 256  # Default height window
x, y = 1100, 1000  # Default center coordinates for the crop window

# List of directories containing images
# Natural sorting function for image filenames
def natural_sort_key(file_path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', file_path)]

'''


input_dirs = [
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Raw Frames/MN_001",
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Raw Frames/MN_002",
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Raw Frames/MN_003",
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Raw Frames/MN_451",
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Raw Frames/MN_452",
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Raw Frames/MN_453",
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Raw Frames/MN_901",
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Raw Frames/MN_902",
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Raw Frames/MN_903",
]
'''
input_dirs = [
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Raw Frames/MN_001",
]

# Define the output directory for collected images
output_base_dir = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/To be processed"
os.makedirs(output_base_dir, exist_ok=True)  # Create the output directory if it doesn't exist
# Run the cropping function for the current directory
choose_window(input_dirs, output_base_dir, x, y, Nx, Ny)




# Call the Data Augmentation function to make the frames more different from each other
# and avoid overfitting
input_dir_augment = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/To be processed"
output_dir_augment = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/AugmentedFrames"
data_augmentation(input_dir_augment, output_dir_augment)




# Adds a mask of Gaussian patches to feed to the Algorithm which will try to remove it and restore the image
input_dir_noisy = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/AugmentedFrames"
output_dir_noisy = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/All processed images"
apply_noise_to_directory(input_dir_noisy, output_dir_noisy, 8, 0.6)



# Paths to your data
input_dir = '/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/All processed images'
target_dir = '/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/AugmentedFrames'

# Load images
input_images, target_images = load_images(input_dir, target_dir, target_size=(32,256))

# Initialize the autoencoder model
input_shape = input_images.shape[1:]  # (height, width, channels)
autoencoder = create_autoencoder(input_shape)

# Train the autoencoder
epochs = 300
batch_size = 32
autoencoder.fit(input_images, target_images,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_split=0.1)

# Save the trained model
autoencoder.save('denoising_autoencoder.h5')
print("Model saved as 'denoising_autoencoder.h5'")



#autoencoder = load_model('/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Codes/SemesterProj/Autoencoder/denoising_autoencoder.h5')
#regularity_scores = evaluate_autoencoder(autoencoder, input_images, target_images)




