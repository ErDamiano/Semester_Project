import cv2
import numpy as np
import os
import glob

def add_noise_patch_to_image(image, patch_size, transparency, output_dir=None, index=0, blur_kernel=(15, 15)):
    """
    Adds a smoothed, upscaled random noise pattern to the image.

    Parameters:
    - image: Input image (grayscale or color).
    - patch_size: Size of the noise patch (default is 16x16).
    - scale_factor: Controls the intensity of the noise overlay.
    - output_dir: Directory to save the noise patch if provided.
    - index: Index number to uniquely save the noise patch.
    - blur_kernel: Tuple specifying the kernel size for Gaussian blur.

    Returns:
    - Noisy image.
    """
    # Step 1: Generate a unique 16x16 noise patch
    noise_patch = np.random.randn(patch_size, patch_size) * 255  # Adjust scale if needed
    noise_patch = noise_patch.astype(np.uint8)  # Convert to 8-bit

    # Save the 16x16 noise patch to the output directory before scaling
    if output_dir:
        patch_path = os.path.join(output_dir, f"noise_patch_{index}.jpg")
        cv2.imwrite(patch_path, noise_patch)
        print(f"Noise patch saved at: {patch_path}")

    # Step 2: Upsample noise to match image size
    noise_large = cv2.resize(noise_patch, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Step 3: Smooth the upsampled noise with Gaussian blur
    noise_large = cv2.GaussianBlur(noise_large, blur_kernel, 0)  # Apply Gaussian blur

    # Save the smoothed noise for inspection (optional)
    if output_dir:
        smoothed_path = os.path.join(output_dir, f"smoothed_noise_{index}.jpg")
        cv2.imwrite(smoothed_path, noise_large)
        print(f"Smoothed noise patch saved at: {smoothed_path}")

    # Step 4: Convert the original image to float32 for compatibility
    image = image.astype(np.float32)
    noise_large = noise_large.astype(np.float32)

    # Step 5: Add noise to image with masking
    noisy_image = cv2.addWeighted(image, 1.0, noise_large, transparency, 0)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)  # Convert back to uint8

    return noisy_image

def apply_noise_to_directory(input_dir, output_dir, patch_size, transparency):
    """
    Applies random noise to all images in the input directory and saves them to the output directory,
    also saves the 16x16 noise patch applied to each image.

    Parameters:
    - input_dir: Directory containing the images to process.
    - output_dir: Directory where the noisy images and patches will be saved.
    - patch_size: Size of the noise patch.
    - scale_factor: Intensity of the noise overlay.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image in the input directory
    for i, image_path in enumerate(glob.glob(os.path.join(input_dir, "*.jpg"))):  # Adjust extension if needed
        # Read image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply noise to the image, and save the 16x16 patch as well
        noisy_image = add_noise_patch_to_image(image, patch_size, transparency, output_dir, index=i)

        # Save noisy image to the output directory
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, noisy_image)
        print(f"Noisy image saved at: {output_path}")

    print(f"Noise added to images in '{input_dir}' and saved to '{output_dir}'.")