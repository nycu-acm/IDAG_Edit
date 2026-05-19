import os
import cv2
import numpy as np
import argparse
from PIL import Image

def convert_and_copy_masks(src_folder, dest_folder, index_offset, start_frame, end_frame):
    # Ensure that the destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Iterate over all files in the source folder
    for filename in os.listdir(src_folder):
        if filename.endswith(".png"):  # Process only PNG files
            # Extract the numeric part of the file name
            file_index = int(filename.split('.')[0])

            # Check if the file index is within the specified frame range
            if start_frame <= file_index <= end_frame:
                # Calculate the new index
                new_index = file_index + index_offset
                # Construct the new file name
                new_filename = f"{new_index:05d}.png"

                # Read the original mask image (grayscale)
                src_path = os.path.join(src_folder, filename)
                mask = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)

                # Apply thresholding to convert the mask to 0 and 1
                _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)

                # Convert mask value range to 0-255
                mask = (mask * 255).astype(np.uint8)

                # Save the converted mask to the destination folder
                dest_path = os.path.join(dest_folder, new_filename)
                cv2.imwrite(dest_path, mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert and copy mask images with index adjustment")
    parser.add_argument('--src_folder', type=str, required=True, help="Path to the source folder containing mask images")
    parser.add_argument('--dest_folder', type=str, required=True, help="Path to the destination folder for converted masks")
    parser.add_argument('--index_offset', type=int, default=0, help="Index offset to apply to file names")
    parser.add_argument('--start_frame', type=int, default=0, help="Start frame number")
    parser.add_argument('--end_frame', type=int, required=True, help="End frame number")

    args = parser.parse_args()

    convert_and_copy_masks(args.src_folder, args.dest_folder, args.index_offset, args.start_frame, args.end_frame)
    print("Mask conversion and copying completed")
