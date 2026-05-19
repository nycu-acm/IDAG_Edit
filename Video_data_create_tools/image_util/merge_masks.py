import cv2
import numpy as np

def merge_two_masks(mask1_path, mask2_path, output_path, method='or'):
    """
    Merge two binary or grayscale masks into a single binary output mask.

    Parameters:
    - mask1_path: path to the first mask image (binary or grayscale).
    - mask2_path: path to the second mask image.
    - output_path: path to save the merged mask image.
    - method: merging operation: 'or', 'and', or 'xor'.
    """
    # Load masks in grayscale
    m1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
    m2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)

    if m1 is None or m2 is None:
        raise FileNotFoundError('Cannot load one or both mask images.')

    # Resize second mask to match first if needed
    if m1.shape != m2.shape:
        m2 = cv2.resize(m2, (m1.shape[1], m1.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Binarize both masks (0 or 255)
    _, b1 = cv2.threshold(m1, 127, 255, cv2.THRESH_BINARY)
    _, b2 = cv2.threshold(m2, 127, 255, cv2.THRESH_BINARY)

    # Merge using bitwise operations
    if method == 'or':
        merged = cv2.bitwise_or(b1, b2)
    elif method == 'and':
        merged = cv2.bitwise_and(b1, b2)
    elif method == 'xor':
        merged = cv2.bitwise_xor(b1, b2)
    else:
        raise ValueError("Unsupported method: choose 'or', 'and', or 'xor'.")

    # Save result
    cv2.imwrite(output_path, merged)
    print(f"Merged mask saved to {output_path}")


if __name__ == '__main__':
    # Example usage:
    merge_two_masks('layout_masks/right_man/00021.png', 'layout_masks/left_man/00021.png', 'merged_or.png', method='or')