"""Auto-calibration: collect labeled square images from a known position.

The starting position contains all 13 classes (every piece type, both
colors, plus empty squares), so a single capture of a board set up for a
new game yields 64 correctly labeled training images of the current chess
set under the current lighting - no manual labeling needed.

Images are saved in the same folder-per-class layout as dataset_v2, so the
collected data can be merged into training directly.
"""

import datetime
import os

import cv2

from board_state import STARTING_PREDICTIONS

CALIBRATION_DIR = "calibration_data"


def save_starting_position_squares(squares, directory=CALIBRATION_DIR):
    """Saves 64 square crops, labeled as the standard starting position.

    Only call this when the physical board actually shows the starting
    position - the labels are assumed, not predicted.
    Returns the number of images written.
    """
    if len(squares) != 64:
        raise ValueError(f"Expected 64 squares, got {len(squares)}")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    count = 0
    for i, (square, label) in enumerate(zip(squares, STARTING_PREDICTIONS)):
        class_dir = os.path.join(directory, label)
        os.makedirs(class_dir, exist_ok=True)
        if cv2.imwrite(os.path.join(class_dir, f"{stamp}_{i:02d}.jpg"), square):
            count += 1
    return count
