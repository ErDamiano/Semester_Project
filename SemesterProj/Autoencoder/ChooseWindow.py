import cv2
import glob
import os
import re

def choose_window(input_dir, output_dir, x, y, Nx, Ny):
    """
    Main function to select a window and crop all images in the input directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the first image for initial window selection
    first_image_path = sorted(glob.glob(f"{input_dir}/*.jpg"))[0]
    first_image = cv2.imread(first_image_path)
    height, width = first_image.shape[:2]

    # Confirm window selection with the user
    while True:
        x1 = max(x - Nx // 2, 0)
        y1 = max(y - Ny // 2, 0)
        x2 = min(x + Nx // 2, width)
        y2 = min(y + Ny // 2, height)

        if display_and_confirm_window(first_image, x1, y1, x2 - x1, y2 - y1):
            break
        else:
            # If not confirmed, prompt user to re-enter x, y, and N
            x = int(input("Enter new x coordinate: "))
            y = int(input("Enter new y coordinate: "))
            Nx = int(input("Enter new window size Nx: "))
            Ny = int(input("Enter new window size Ny: "))

    # Process each image in the directory
    for i, image_path in enumerate(sorted(glob.glob(f"{input_dir}/*.jpg"), key=natural_sort_key)):
        img = cv2.imread(image_path)

        # Crop the selected window
        cropped_img = img[y1:y2, x1:x2]

        # Save the cropped image to the output directory
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"cropped_{filename}")  # Prefix or change the filename as needed
        cv2.imwrite(output_path, cropped_img)

    print("All images cropped and saved.")

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