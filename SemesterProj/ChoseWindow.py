import cv2
import glob
import os
import re

# Set initial parameters
Nx = 256  # Default width window
Ny = 256 # Default height window
x, y = 1200, 1200  # Default center coordinates for the crop window

# Input and output directories for MAC
input_dir = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/stabilizedFrames/MN_001"
output_dir = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/croppedFrames/MN_001"
os.makedirs(output_dir, exist_ok=True)


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


# Read the first image for initial window selection
first_image_path = sorted(glob.glob(f"{input_dir}/*.jpg"))[0]
first_image = cv2.imread(first_image_path)
height, width = first_image.shape[:2]

# Initial window selection and confirmation
while True:
    x1 = max(x - Nx // 2, 0)
    y1 = max(y - Ny // 2, 0)
    x2 = min(x + Nx // 2, width)
    y2 = min(y + Ny // 2, height)

    # Confirm window selection with the user
    if display_and_confirm_window(first_image, x1, y1, x2 - x1, y2 - y1):
        break
    else:
        # If not confirmed, prompt user to re-enter x, y, and N
        x = int(input("Enter new x coordinate: "))
        y = int(input("Enter new y coordinate: "))
        Nx = int(input("Enter new window size Nx: "))
        Ny = int(input("Enter new window size Ny: "))

# Sort images
def natural_sort_key(file_path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', file_path)]
image_files = sorted(glob.glob(f"{input_dir}/*.jpg"), key=natural_sort_key)
# Process all images with the confirmed window
for i, file_path in enumerate(image_files):
    # Read each image
    image = cv2.imread(file_path)

    # Crop and save each image based on confirmed window parameters
    cropped_image = image[y1:y2, x1:x2]
    output_path = os.path.join(output_dir, f"cropped_frame_{i}.jpg")
    cv2.imwrite(output_path, cropped_image)
    print(f"Cropped image saved as {output_path}")

print("Cropping completed for all frames and saved to:", output_dir)
