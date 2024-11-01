import cv2
import numpy as np
import glob

# Load frames Mac
frames = [cv2.imread(file) for file in sorted(glob.glob("/Volumes/MEMORIAEST/SemesterProjectData/2024_01_08E/MN_90/2024_01_08E_MN_90_003/pics/2D/*.tiff"))]
output_folder = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/stabilizedFrames/MN_903/"
# Load frames Windows
# frames = [cv2.imread(file) for file in sorted(glob.glob("G:\2D\2024_01_08E_PU_00_001-0001_0.tiff"))]

# Convert frames to grayscale for better feature detection
gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

# Define the first frame as the reference
reference_frame = gray_frames[0]
height, width = reference_frame.shape

# Allow the user to select a ROI for stabilization
roi = cv2.selectROI("Select ROI", reference_frame)
x, y, w, h = roi

# Create an empty list to hold stabilized frames
stabilized_frames = []

# Loop through each frame and align to the reference
for frame in gray_frames:
    # Detect ORB keypoints and descriptors within the ROI
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(reference_frame[y:y+h, x:x+w], None)
    kp2, des2 = orb.detectAndCompute(frame[y:y+h, x:x+w], None)

    # Match features between the reference frame and current frame
    if des1 is not None and des2 is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched keypoints
        if len(matches) >= 3:  # Ensure there are enough matches
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate transformation matrix
            matrix, mask = cv2.estimateAffinePartial2D(pts2, pts1)

            # Check if matrix is valid
            if matrix is not None:
                # Apply affine transformation to stabilize the frame
                stabilized_frame = cv2.warpAffine(frame, matrix, (width, height))
            else:
                stabilized_frame = frame  # If no valid matrix, keep original frame
        else:
            stabilized_frame = frame  # Not enough matches, keep original frame
    else:
        stabilized_frame = frame  # No descriptors found, keep original frame

    # Append stabilized frame to the list
    stabilized_frames.append(stabilized_frame)

# Save or display stabilized frames
for i, frame in enumerate(stabilized_frames):
    cv2.imwrite(f"{output_folder}stabilized_frame_{i}.jpg", frame)
    print(f"Saved frame {i}")

# Close the ROI selection window
cv2.destroyAllWindows()