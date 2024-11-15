import os
import glob
import cv2
import numpy as np
import re

def natural_sort_key(s):
    """Custom sort function to handle numeric sorting in file names."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def load_clips(input_dir, Nx, Ny, frames_per_clip):
    """
    Loads clips from subfolders in input_dir, transforms each into an input cuboid of
    shape (frames_per_clip - 1, Nx, Ny, 1) and a target image of shape (Nx, Ny, 1).

    Parameters:
        input_dir (str): Directory containing subfolders with each clip's frames.
        Nx (int): Width of each frame.
        Ny (int): Height of each frame.
        frames_per_clip (int): Number of frames per clip, with the last frame as the target.

    Returns:
        input_cuboids (np.ndarray): Array of input cuboids for the autoencoder.
        target_images (np.ndarray): Array of target images for each cuboid.
    """
    input_cuboids = []
    target_images = []

    # Iterate through each clip's folder in the input directory
    for clip_folder in sorted(os.listdir(input_dir), key=natural_sort_key):  # Sorting with the custom key
        clip_path = os.path.join(input_dir, clip_folder)
        if os.path.isdir(clip_path):
            # Retrieve all image paths in the subfolder and sort them naturally
            image_paths = sorted(glob.glob(os.path.join(clip_path, "*.jpg")), key=natural_sort_key)

            # Check if the number of frames matches the expected count
            if len(image_paths) < frames_per_clip:
                print(f"Warning: {clip_folder} has fewer than {frames_per_clip} frames, skipping.")
                continue

            # Load frames and resize to (Nx, Ny) if necessary
            frames = []
            for img_path in image_paths[:frames_per_clip - 1]:  # Load all but the last frame
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img.shape[:2] != (Ny, Nx):
                    img = cv2.resize(img, (Nx, Ny))
                frames.append(img)

            # Stack frames into a hyper-cuboid of shape (frames_per_clip - 1, Nx, Ny, 1)
            cuboid = np.stack(frames, axis=0)
            cuboid = np.expand_dims(cuboid, axis=-1)  # Add channel dimension

            # Load and resize the target image (last frame in the clip)
            target_img_path = image_paths[frames_per_clip - 1]  # Correct indexing to get the last frame
            target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
            if target_img.shape[:2] != (Ny, Nx):
                target_img = cv2.resize(target_img, (Nx, Ny))
            target_img = np.expand_dims(target_img, axis=-1)  # Add channel dimension

            # Print the name of the target image
            print(f"Target image created: {target_img_path}")

            # Append the cuboid and target image to the respective lists
            input_cuboids.append(cuboid)
            target_images.append(target_img)

    # Convert lists to numpy arrays once after the loop
    input_cuboids = np.array(input_cuboids, dtype=np.float32) / 255.0  # Normalize pixel values
    target_images = np.array(target_images, dtype=np.float32) / 255.0

    print(f"Loaded {len(input_cuboids)} cuboids from {input_dir}.")
    return input_cuboids, target_images





