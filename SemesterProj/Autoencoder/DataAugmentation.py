import cv2
import glob
import imgaug.augmenters as iaa
import numpy as np
import os

def data_augmentation(input_dir, output_dir):

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect all file paths from the input directory
    all_files = glob.glob(os.path.join(input_dir, "*.jpg"))  # Adjust extension if needed

    # Load images in grayscale
    images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in all_files]
    # Convert images to a format suitable for imgaug
    images = np.array(images)

    # Define an augmentation sequence
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # Flip images horizontally with 50% probability
        iaa.Affine(
            rotate=(-10, 10),  # Rotate images by -10 to +10 degrees
            scale=(0.9, 1.1)  # Scale images to 90% to 110%
        ),
        iaa.AdditiveGaussianNoise(scale=(0, 0.01 * 255)),  # Add Gaussian noise
        iaa.LinearContrast((0.8, 1.2)),  # Adjust contrast
        iaa.Crop(percent=(0, 0.1)),  # Randomly crop images
    ])

    # Apply augmentations to each image
    augmented_images = seq(images=images)

    # Save augmented images
    for i, img in enumerate(augmented_images):
        img = np.clip(img, 0, 255).astype(np.uint8)  # Ensure the image is in the correct format
        cv2.imwrite(f"{output_dir}/augmented_image_{i}.jpg", img)

    print("Augmentation completed and saved to:", output_dir)