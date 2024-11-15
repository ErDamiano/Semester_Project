import cv2
import glob
import os
import re

def choose_window(input_dirs, output_base_dir, x, y, Nx, Ny, Num_crops_per_image):
    """
    Function to crop five adjacent windows from each image in multiple input directories,
    saving them sequentially in a single output folder with subframe numbering.
    """
    # Ensure the base output directory exists
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    image_count = 1  # Counter for the original image
    confirmed = False  # Flag to track window confirmation

    # Iterate over each input directory
    for input_dir in input_dirs:
        # List all .tiff and .tif files in the directory
        image_paths = sorted(glob.glob(f"{input_dir}/*.tiff")) + \
                      sorted(glob.glob(f"{input_dir}/*.tif")) + \
                      sorted(glob.glob(f"{input_dir}/*.png"))

        # Confirm at least one image exists
        if not image_paths:
            print(f"No images found in {input_dir}. Skipping this directory.")
            continue

        # Crop five strips from each image
        for image_path in image_paths:
            img = cv2.imread(image_path)
            height, width = img.shape[:2]

            # Display and confirm window selection only on the first image
            if not confirmed:
                if not display_and_confirm_window(img, x, y, Nx, Ny):
                    print("Exiting. Adjust the coordinates and try again.")
                    return
                confirmed = True  # Set confirmation flag to True after first image

            # Adjust starting x position for the first strip
            x_start = max(x - Nx // 2, 0)

            # Crop and save five strips
            for subframe_index in range(Num_crops_per_image):
                x1 = x_start + subframe_index * Nx
                x2 = x1 + Nx

                # Ensure the crop does not exceed the image width
                if x2 > width:
                    print(f"Subframe {subframe_index + 1} for image {image_count} exceeds width. Skipping this strip.")
                    continue

                # Crop the strip
                cropped_img = img[y:y + Ny, x1:x2]

                # Save with the format imagecount_subframeindex.jpg (e.g., 1_1, 1_2, etc.)
                output_path = os.path.join(output_base_dir, f"{image_count}_{subframe_index + 1}.jpg")
                cv2.imwrite(output_path, cropped_img)

            # Increment the original image count after processing five strips
            image_count += 1

    print(f"All images cropped into five strips and saved sequentially in {output_base_dir}.")

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


