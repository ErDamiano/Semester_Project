import os
import re
import glob
import cv2
import tensorflow as tf
from tensorflow.python.keras.saving.save import load_model
import numpy as np

from ClipGenerator import ClipGenerator
from ChooseWindowClip import choose_window, display_and_confirm_window
from ClipAutoEncoder import build_convlstm_model
from LoadClips import load_clips


# Set initial parameters to crop the images around x y with a window size of Nx and Ny
Nx = 32  # Default width window
Ny = 128  # Default height window
x, y = 1100, 970  # Default center coordinates for the crop window
num_crops_per_image = 1
frames_per_clip = 8

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
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Raw Frames/PU_001",
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Raw Frames/PU_002",
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Raw Frames/PU_003",
]

'''


input_dirs_clips = [
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Raw Frames/MN_002",
]
output_dir_clips = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Raw Clips"
os.makedirs(output_dir_clips, exist_ok=True)  # Create the output directory if it doesn't exist

ClipGenerator(input_dirs_clips, output_dir_clips, frames_per_clip, num_crops_per_image)



# Set input and output directories
input_dir_window = output_dir_clips
output_dir_window = '/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Cropped Clips'



# Get list of subfolders in input_dir
subfolders = [os.path.join(input_dir_window, folder) for folder in sorted(os.listdir(input_dir_window)) if os.path.isdir(os.path.join(input_dir_window, folder))]
# Apply the confirmed coordinates to all subfolders without further confirmation
choose_window(subfolders, output_dir_window, x, y, Nx, Ny, num_crops_per_image)

'''
# Paths to your data for Autoencoder
input_dir = '/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Cropped Clips'


inputs, targets = load_clips(input_dir, Nx,Ny, frames_per_clip)

# Check shapes
print(f"Inputs shape: {inputs.shape}")
print(f"Targets shape: {targets.shape}")

input_shape = (frames_per_clip-1, Ny, Nx, 1)
model = build_convlstm_model(input_shape)
model.fit(inputs, targets, batch_size=4, epochs=100, validation_split=0.1)

model.save('Predicting.h5')
print("Model saved as 'Predicting.h5'")

'''

