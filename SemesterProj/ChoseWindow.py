import cv2
import glob
import os

# Set parameters
N = 256  # Size of the NxN window
x, y = 1200, 1200  # Center coordinates for the crop window

# Input and output directories
input_dir = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/stabilizedFrames/"
output_dir = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/croppedFrames/"
os.makedirs(output_dir, exist_ok=True)

# Process each image
for i, file_path in enumerate(sorted(glob.glob(f"{input_dir}/*.jpg"))):
    # Read the image
    image = cv2.imread(file_path)
    height, width = image.shape[:2]

    # Calculate crop boundaries, ensuring they are within image limits
    x1 = max(x - N // 2, 0)
    y1 = max(y - N // 2, 0)
    x2 = min(x + N // 2, width)
    y2 = min(y + N // 2, height)

    # Crop and save the image
    cropped_image = image[y1:y2, x1:x2]
    output_path = os.path.join(output_dir, f"cropped_frame_{i}.jpg")
    cv2.imwrite(output_path, cropped_image)

print("Cropping completed and saved to:", output_dir)

