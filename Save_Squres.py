import cv2
import numpy as np
import os

import time




# Load the image
image_path = "imgs/topdown_board.png"  # Update this to your image path
image = cv2.imread(image_path)
if image is None:
    print("Error loading image.")
    exit()

# Define destination for saved squares
save_dir = "chessboard_squares"
os.makedirs(save_dir, exist_ok=True)


# Function to manually select four points for perspective transformation
def select_points(image):
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                print(f"Point selected: ({x}, {y})")
                if len(points) == 4:
                    cv2.destroyWindow("Select Points")

    cv2.imshow("Select Points", image)
    cv2.setMouseCallback("Select Points", mouse_callback)

    # Wait until 4 points are selected
    while len(points) < 4:
        cv2.waitKey(1)

    return np.array(points, dtype="float32")


# Perspective transform function
def perspective_transform(image, src_pts):
    dst_pts = np.array([[0, 0], [500, 0], [500, 500], [0, 500]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, M, (500, 500))


# Function to split and save each chessboard square
def save_squares(warped_image, board_size=8):
    height, width = warped_image.shape[:2]
    square_height = height // board_size
    square_width = width // board_size

    for row in range(board_size):
        for col in range(board_size):
            start_x = col * square_width
            start_y = row * square_height
            end_x = (col + 1) * square_width
            end_y = (row + 1) * square_height

            square = warped_image[start_y:end_y, start_x:end_x]
            timestamp = int(time.time())  # Get current time in seconds since epoch
            filename = os.path.join(save_dir, f"square_{row}_{col}_{timestamp}.png")
            cv2.imwrite(filename, square)
            print(f"Saved {filename}")


# Select points, transform perspective, and save squares
src_points = select_points(image)
warped_image = perspective_transform(image, src_points)
save_squares(warped_image)
print(f"All squares saved to '{save_dir}'")
