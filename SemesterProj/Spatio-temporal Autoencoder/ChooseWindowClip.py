import cv2
import glob
import os
import re

def choose_window(input_dirs, output_base_dir, x, y, Nx, Ny, Num_crops_per_image):
    """
    Function to crop a window from each image in multiple input directories
    and save them as sequentially named clips without further subfolder nesting.
    """
    # Ensure the base output directory exists
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    confirmed = False  # Flag to track window confirmation
    clip_count = 1  # To track the total number of clips

    # Iterate over each input directory, which corresponds to each raw clip
    for input_dir in input_dirs:
        # List all image files (e.g., .png) in the directory
        image_paths = sorted(glob.glob(f"{input_dir}/*.png"), key=natural_sort_key)

        # Confirm at least one image exists
        if not image_paths:
            print(f"No images found in {input_dir}. Skipping this directory.")
            continue

        # Load the first image and confirm window selection only once
        if not confirmed:
            first_image = cv2.imread(image_paths[0])
            if not display_and_confirm_window(first_image, x, y, Nx, Ny):
                print("Exiting. Adjust the coordinates and try again.")
                return
            confirmed = True  # Set confirmation flag to True after first image

        # Generate multiple cropped clips based on `Num_crops_per_image`
        for crop_index in range(Num_crops_per_image):
            # Create a unique subfolder for each cropped clip
            output_clip_dir = os.path.join(output_base_dir, f"cropped_clip_{clip_count}")
            os.makedirs(output_clip_dir, exist_ok=True)

            # Crop and save each frame in the current clip
            for frame_num, image_path in enumerate(image_paths, start=1):
                img = cv2.imread(image_path)
                height, width = img.shape[:2]

                # Adjust starting x position based on crop index
                x1 = max(x - Nx // 2 + crop_index * Nx, 0)
                x2 = x1 + Nx

                # Ensure the crop does not exceed the image width
                if x2 > width:
                    print(f"Frame {frame_num} in {input_dir} exceeds width. Skipping this frame.")
                    continue

                # Crop the frame
                cropped_img = img[y:y + Ny, x1:x2]

                # Save each frame in sequence (1.jpg, 2.jpg, ..., 10.jpg)
                output_path = os.path.join(output_clip_dir, f"{frame_num}.jpg")
                cv2.imwrite(output_path, cropped_img)

            # Increment clip count after processing this set of crops
            clip_count += 1

    print(f"All images cropped and saved in {output_base_dir}.")

# Natural sorting function for image filenames
def natural_sort_key(file_path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', file_path)]

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
    choice = input("Do you want to use this window for all frames? (y to proceed / no to change): ").strip().lower()
    cv2.destroyWindow("Window Selection")

    return choice == "y"





