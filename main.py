import cv2
import numpy as np

from models.square_classifier import load_model, predict_square
from board_utils import generate_digital_board, perspective_transform, split_into_squares
from board_state import predictions_to_placement

IMAGE_PATH = "imgs/test2.JPG"

# Hardcoded labels for a standard starting position (debug mode)
DEBUG_PREDS = (
    ["br", "bn", "bb", "bq", "bk", "bb", "bn", "br"]
    + ["bp"] * 8
    + ["empty"] * 32
    + ["wp"] * 8
    + ["wr", "wn", "wb", "wq", "wk", "wb", "wn", "wr"]
)

points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Point selected: ({x}, {y})")


def main():
    model = load_model("models/weights.pth")

    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Error loading image: {IMAGE_PATH}")
        return

    cv2.namedWindow("Chessboard")
    cv2.setMouseCallback("Chessboard", mouse_callback)

    print("--- STATIC IMAGE MODE ---")
    print("1. Click 4 corners of the board (top-left first, clockwise).")
    print("2. Press 'p' to predict, 'd' for a debug starting position, 'q' to quit.")

    while True:
        img_copy = image.copy()

        for point in points:
            cv2.circle(img_copy, point, 5, (0, 255, 0), -1)
        if len(points) == 4:
            cv2.polylines(img_copy, [np.array(points)], isClosed=True, color=(255, 0, 0), thickness=2)

        cv2.imshow("Chessboard", img_copy)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('p') and len(points) == 4:
            src_pts = np.array(points, dtype="float32")
            warped = perspective_transform(image, src_pts)

            print("Predicting...")
            preds = [predict_square(model, sq) for sq in split_into_squares(warped)]
            print(f"FEN: {predictions_to_placement(preds)}")

            cv2.imshow("Digital State", generate_digital_board(preds))

        if key == ord('d'):
            print("Debug Mode: Showing standard starting position...")
            print(f"FEN: {predictions_to_placement(DEBUG_PREDS)}")
            cv2.imshow("Digital State", generate_digital_board(DEBUG_PREDS))

        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
