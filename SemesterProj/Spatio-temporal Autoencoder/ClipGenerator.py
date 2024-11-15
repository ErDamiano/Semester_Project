import os
import cv2
import glob
from PIL import Image
import re

def ClipGenerator(input_dirs, output_dir, x, crops):
    """
    Generates clips of specified frame intervals from images in the input directory.
    Discards the first few frames if total frames are not divisible by x.

    Parameters:
    - input_dirs: List of directories containing input images.
    - output_dir_clips: Directory to save the generated clips.
    - x: Number of frames in each clip.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for input_dir in input_dirs:
        # List all .tiff and .tif files in the directory
        image_paths = sorted(glob.glob(f"{input_dir}/*.tiff")) + \
                      sorted(glob.glob(f"{input_dir}/*.tif"))
        num_images = len(image_paths)

        # Confirm at least x images exist
        if num_images < x:
            print(f"Not enough images in {input_dir} for a single clip. Skipping this directory.")
            continue

        # Discard the first few frames to make the total divisible by x
        frames_to_discard = num_images % x
        if frames_to_discard > 0:
            image_paths = image_paths[frames_to_discard:]  # Skip the first few frames
            num_images = len(image_paths)

        # Calculate the number of clips to create
        num_clips = num_images // x

        # Create clips
        for i in range(num_clips):
            clip_frames = []

            # Collect x frames for the current clip, starting with the (i+1)th frame and jumping by x frames
            for j in range(x):
                frame_idx = i + j * num_clips
                clip_frames.append(image_paths[frame_idx])

            # Save the frames for the clip into a new directory
            clip_dir = os.path.join(output_dir, f'clip_{i + 1}')
            os.makedirs(clip_dir, exist_ok=True)

            for k, frame_path in enumerate(clip_frames):
                frame = Image.open(frame_path)
                frame.save(os.path.join(clip_dir, f'{k + 1}.png'))

    print(f"{num_clips*crops} clips created, each containing {x} frames.")

def natural_sort_key(file_path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', file_path)]
