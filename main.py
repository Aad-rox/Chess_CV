import cv2
import numpy as np

# Initialize global variables for storing points
points = []


# Mouse callback function to capture the four corner points
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Point selected: ({x}, {y})")
            if len(points) == 4:
                print("Four points selected. Press 'p' to perform perspective transform or 'q' to quit.")


# Function to detect chessboard corners for an 8x8 pattern
def detect_chessboard(image, pattern_size=(8, 8)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if found:
        print("Chessboard detected.")
        return corners
    else:
        print("Chessboard not detected.")
        return None


# Function to perform perspective transformation
def perspective_transform(image, src_pts):
    # Define the destination points for the "flattened" image
    dst_pts = np.array([[0, 0], [500, 0], [500, 500], [0, 500]], dtype="float32")

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the warp perspective
    warped = cv2.warpPerspective(image, M, (500, 500))
    return warped

def highlight_chessboard_squares(warped_image, board_size=8):
    # Get the height and width of the warped image
    height, width = warped_image.shape[:2]

    # Calculate the size of each square
    square_height = height // board_size
    square_width = width // board_size

    # Loop through the rows and columns of the chessboard
    for row in range(board_size):
        for col in range(board_size):
            # Calculate the start and end coordinates of each square
            start_x = col * square_width
            start_y = row * square_height
            end_x = (col + 1) * square_width
            end_y = (row + 1) * square_height

            # Draw a rectangle around each square
            cv2.rectangle(warped_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    return warped_image

def split_chessboard_into_squares(warped_image, board_size=8):
    # Get the height and width of the warped image
    height, width = warped_image.shape[:2]

    # Calculate the size of each square
    square_height = height // board_size
    square_width = width // board_size

    squares = []

    # Loop through the rows and columns of the chessboard
    for row in range(board_size):
        for col in range(board_size):
            # Calculate the start and end coordinates of each square
            start_x = col * square_width
            start_y = row * square_height
            end_x = (col + 1) * square_width
            end_y = (row + 1) * square_height

            # Extract the square using slicing
            square = warped_image[start_y:end_y, start_x:end_x]

            # Store each square in the list
            squares.append(square)

            # Optionally display the square (for debugging)
            cv2.imshow(f"Square {row}-{col}", square)
            cv2.waitKey(100)  # Display each square for a short time

    return squares
# Main function
def main():
    # Load chessboard image
    image = cv2.imread("imgs/chess_knight.png")
    if image is None:
        print("Error loading image.")
        return

    # Detect chessboard corners
    corners = detect_chessboard(image)

    # If chessboard is detected, draw the detected corners
    if corners is not None:
        cv2.drawChessboardCorners(image, (8, 8), corners, True)

    # Create a window and set the mouse callback function
    cv2.namedWindow("Chessboard")
    cv2.setMouseCallback("Chessboard", mouse_callback)


    while True:
        # Display the image
        img_copy = image.copy()

        # Mark the selected corners with circles
        for point in points:
            cv2.circle(img_copy, point, 5, (0, 255, 0), -1)

        # If four points are selected, draw a quadrilateral
        if len(points) == 4:
            cv2.polylines(img_copy, [np.array(points)], isClosed=True, color=(255, 0, 0), thickness=2)

        # Show the image
        cv2.imshow("Chessboard", img_copy)

        # Press 'p' to perform perspective transform when four points are selected
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p') and len(points) == 4:
            # Perform perspective transformation
            src_pts = np.array(points, dtype="float32")
            warped_image = perspective_transform(image, src_pts)
            cv2.imshow("Warped Chessboard", warped_image)
            squares=highlight_chessboard_squares(warped_image)
            cv2.imshow("Squares Chessboard", squares)
            all_squares = split_chessboard_into_squares(warped_image)



        # Press 'q' to exit
        if key == ord('q'):
            break

    # Close all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

