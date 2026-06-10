import cv2
import numpy as np
import time

from models.square_classifier import load_model, predict_square
from board_utils import generate_digital_board, perspective_transform, split_into_squares
from board_state import GameTracker

# --- CONFIGURATION ---
PREDICTION_INTERVAL = 2.0  # Updates board every 2 seconds
WEBCAM_ID = 0  # Change to 1 if using an external webcam

# --- LOAD RESOURCES ---
model = load_model("models/weights.pth")

# --- MOUSE LOGIC ---
points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))


# --- MAIN LIVE LOOP ---
def main():
    cap = cv2.VideoCapture(WEBCAM_ID)

    # Try to force High Res (Optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow("Live Feed")
    cv2.setMouseCallback("Live Feed", mouse_callback)

    # State tracking
    last_prediction_time = 0
    current_preds = ["empty"] * 64
    digital_board_img = generate_digital_board(current_preds)
    tracker = GameTracker()

    print("--- LIVE MODE ---")
    print("1. Click 4 corners of the board (white at the bottom).")
    print(f"2. The digital board updates every {PREDICTION_INTERVAL} seconds.")
    print("3. Detected moves and FEN are printed as the game progresses.")
    print("4. Press 'r' to reset corners.")
    print("5. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. UI: Draw Circles/Lines
        display_frame = frame.copy()
        for i, pt in enumerate(points):
            cv2.circle(display_frame, pt, 5, (0, 255, 0), -1)
            cv2.putText(display_frame, str(i + 1), (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if len(points) == 4:
            cv2.polylines(display_frame, [np.array(points)], True, (255, 0, 0), 2)

            # 2. TIME CHECK: Is it time to predict?
            current_time = time.time()
            if current_time - last_prediction_time > PREDICTION_INTERVAL:

                try:
                    src_pts = np.array(points, dtype="float32")
                    warped = perspective_transform(frame, src_pts)

                    current_preds = [predict_square(model, sq)
                                     for sq in split_into_squares(warped)]
                    digital_board_img = generate_digital_board(current_preds)
                    last_prediction_time = current_time

                    # 3. GAME TRACKING: match the readout to a legal move
                    result = tracker.update(current_preds)
                    if result["status"] == "move":
                        move_num = (len(tracker.moves) + 1) // 2
                        print(f"Move detected: {move_num}. {result['san']}")
                        print(f"  FEN: {tracker.fen}")
                    elif result["status"] == "new_game":
                        print("New game detected - tracker reset.")
                    elif result["status"] == "unrecognized":
                        print("Readout doesn't match any legal move "
                              "(noise, hand over board, or missed moves) - ignored.")

                except Exception as e:
                    print(f"Prediction Error: {e}")

        # 4. Show Windows
        cv2.imshow("Live Feed", display_frame)
        cv2.imshow("Digital Board", digital_board_img)

        # 5. Input Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            points.clear()
            print("Points reset.")
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print the final game record
    if tracker.moves:
        print("\nGame moves:", " ".join(tracker.moves))
        print(f"Final FEN: {tracker.fen}")


if __name__ == "__main__":
    main()


