import cv2
import glob
import os
import numpy as np

def load_images(input_dir, target_dir, target_size):
    """
    Loads, resizes, and pairs input (noisy) and target (clean) images for training.

    Parameters:
    - input_dir: Directory containing noisy input images.
    - target_dir: Directory containing clean target images.
    - target_size: Tuple specifying the desired image dimensions, e.g., (128, 128).

    Returns:
    - input_images: Array of resized and normalized noisy input images with added channel dimension.
    - target_images: Array of resized and normalized clean target images with added channel dimension.
    """
    input_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    target_paths = glob.glob(os.path.join(target_dir, "*.jpg"))

    # Sort paths by the numeric part of the filename
    input_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    target_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    if len(input_paths) != len(target_paths):
        raise ValueError("The number of input and target images must be the same")

    input_images = []
    target_images = []

    for input_path, target_path in zip(input_paths, target_paths):
        input_img = cv2.resize(cv2.imread(input_path, cv2.IMREAD_GRAYSCALE), target_size)
        target_img = cv2.resize(cv2.imread(target_path, cv2.IMREAD_GRAYSCALE), target_size)

        input_images.append(input_img)
        target_images.append(target_img)

    # Normalize images
    input_images = np.array(input_images).astype('float32') / 255.0
    target_images = np.array(target_images).astype('float32') / 255.0

    # Add a channel dimension for grayscale images
    input_images = np.expand_dims(input_images, axis=-1)
    target_images = np.expand_dims(target_images, axis=-1)

    return input_images, target_images