import cv2
import glob
import os
import re

def choose_window(input_dirs, output_base_dir, x, y, Nx, Ny):
    """
    Main function to select a window and crop all images in multiple input directories,
    saving them sequentially in a single output folder.
    """
    # Ensure the base output directory exists
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    image_count = 1  # Initialize count for sequential numbering

    # Iterate over each input directory
    for input_dir in input_dirs:
        # List all .tiff, .TIFF, and .tif files in the directory
        image_paths = sorted(glob.glob(f"{input_dir}/*.tiff")) + \
                      sorted(glob.glob(f"{input_dir}/*.TIFF")) + \
                      sorted(glob.glob(f"{input_dir}/*.tif"))

        # Confirm at least one image exists
        if not image_paths:
            print(f"No images found in {input_dir}. Skipping this directory.")
            continue

        # Read the first image to determine the cropping window
        first_image = cv2.imread(image_paths[0])
        height, width = first_image.shape[:2]

        # Define crop coordinates
        x1 = max(x - Nx // 2, 0)
        y1 = max(y - Ny // 2, 0)
        x2 = min(x + Nx // 2, width)
        y2 = min(y + Ny // 2, height)

        # Crop and save each image in the output directory with sequential numbering
        for image_path in image_paths:
            img = cv2.imread(image_path)
            cropped_img = img[y1:y2, x1:x2]

            # Save cropped image to output directory with sequential numbering
            output_path = os.path.join(output_base_dir, f"{image_count}.jpg")
            cv2.imwrite(output_path, cropped_img)
            image_count += 1

    print(f"All images cropped and saved sequentially in {output_base_dir}.")

# Function to display and confirm the window selection on the first image
def display_and_confirm_window(image, x, y, width, height):
    # Create a copy of the image to draw the overlay
    image_copy = image.copy()

    # Draw a red rectangle on the image as a window overlay
    color = (0, 0, 255)  # Red color in BGR
    thickness = 2
    cv2.rectangle(image_copy, (x, y), (x + width, y + height), color, thickness)

    # Display the image with the overlay
    cv2.imshow("Window Selection", image_copy)
    cv2.waitKey(1)  # Small delay to render the window

    # Ask the user to confirm
    choice = input("Do you want to use this window for all frames? (yes to proceed / no to change): ").strip().lower()
    cv2.destroyWindow("Window Selection")

    return choice == "yes"

# Natural sorting function for image filenames
def natural_sort_key(file_path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', file_path)]