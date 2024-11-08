import cv2
import os

def crop_images(input_folder, output_folder, x, y, width, height):
    """
    Crop images from the input folder and save them to the output folder.

    Parameters:
        input_folder (str): The path to the folder containing images.
        output_folder (str): The path to the folder to save cropped images.
        x (int): The x-coordinate of the top-left corner of the cropping window.
        y (int): The y-coordinate of the top-left corner of the cropping window.
        width (int): The width of the cropping window.
        height (int): The height of the cropping window.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".jpeg", ".png", ".tiff")):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            if image is not None:
                # Ensure the cropping window is within the image dimensions
                x = min(x, image.shape[1] - 1)
                y = min(y, image.shape[0] - 1)
                width = min(width, image.shape[1] - x)
                height = min(height, image.shape[0] - y)

                # Crop the image
                cropped_image = image[y:y + height, x:x + width]

                # Save the cropped image to the output folder
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, cropped_image)
                print(f"Cropped image saved: {output_path}")
            else:
                print(f"Failed to load image: {filename}")
        else:
            print(f"Skipped non-image file: {filename}")

# Example usage
if __name__ == "__main__":
    input_folder = os.path.expanduser("~/Desktop/fotoDaTagliare")  # Change to your input folder
    output_folder = os.path.expanduser("~/Desktop/fotoTagliate")    # Change to your output folder
    x = 500  # Example x-coordinate
    y = 300  # Example y-coordinate
    width = 2000  # Example width
    height = 1500  # Example height

    crop_images(input_folder, output_folder, x, y, width, height)