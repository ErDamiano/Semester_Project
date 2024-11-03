import cv2
import numpy as np
import os
import glob
import re

# Natural sorting function for image filenames
def natural_sort_key(file_path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', file_path)]

def add_noise_patch_to_image(image, patch_size, transparency, output_dir=None, index=0, blur_kernel=(15, 15)):
    """
    Adds a smoothed, upscaled random noise pattern to the image.

    Parameters:
    - image: Input image (grayscale or color).
    - patch_size: Size of the noise patch (default is 16x16).
    - transparency: Controls the intensity of the noise overlay.
    - output_dir: Directory to save the noise patch if provided.
    - index: Index number to uniquely save the noise patch.
    - blur_kernel: Tuple specifying the kernel size for Gaussian blur.

    Returns:
    - Noisy image.
    """
    # Step 1: Generate a unique 16x16 noise patch
    noise_patch = np.random.randn(patch_size, patch_size) * 255  # Adjust scale if needed
    noise_patch = noise_patch.astype(np.uint8)  # Convert to 8-bit

    # Step 2: Upsample noise to match image size
    noise_large = cv2.resize(noise_patch, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Step 3: Smooth the upsampled noise with Gaussian blur
    noise_large = cv2.GaussianBlur(noise_large, blur_kernel, 0)  # Apply Gaussian blur

    # Step 4: Convert the original image to float32 for compatibility
    image = image.astype(np.float32)
    noise_large = noise_large.astype(np.float32)

    # Step 5: Add noise to image with masking
    noisy_image = cv2.addWeighted(image, 1.0, noise_large, transparency, 0)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)  # Convert back to uint8

    return noisy_image

def apply_noise_to_directory(input_dir, output_dir, patch_size, transparency):
    """
    Applies random noise to all images in the input directory and saves them to the output directory.

    Parameters:
    - input_dir: Directory containing the images to process.
    - output_dir: Directory where the noisy images will be saved.
    - patch_size: Size of the noise patch.
    - transparency: Intensity of the noise overlay.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect and sort image paths in natural order
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")), key=natural_sort_key)

    # Process each image in the sorted input directory
    for i, image_path in enumerate(image_paths):
        # Read image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply noise to the image
        noisy_image = add_noise_patch_to_image(image, patch_size, transparency, output_dir, index=i)

        # Save noisy image to the output directory
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, noisy_image)
        print(f"Noisy image saved at: {output_path}")

    print(f"Noise added to images in '{input_dir}' and saved to '{output_dir}'.")