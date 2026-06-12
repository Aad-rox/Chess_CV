import cv2
import numpy as np
import time
from collections import Counter, deque

from models.square_classifier import load_model, predict_board, get_device
from board_utils import generate_digital_board, perspective_transform, split_into_squares
from board_state import GameTracker

# --- CONFIGURATION ---
PREDICTION_INTERVAL = 0.4  # Seconds between board readouts
VOTE_WINDOW = 5  # Each square's label = majority vote over this many readouts
WEBCAM_ID = 0  # Change to 1 if using an external webcam

# --- LOAD RESOURCES ---
model = load_model("models/weights.pth")
model.to(get_device())

# --- MOUSE LOGIC ---
points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))


def majority_vote(histories):
    """Per-square majority label over the recent readout history."""
    return [Counter(h).most_common(1)[0][0] for h in histories]


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
    digital_board_img = generate_digital_board(["empty"] * 64)
    tracker = GameTracker()
    # One rolling history per square; a single bad frame (hand over the
    # board, motion blur) loses the vote instead of corrupting the board.
    histories = [deque(maxlen=VOTE_WINDOW) for _ in range(64)]
    last_warned_placement = None

    print("--- LIVE MODE ---")
    print("1. Click 4 corners of the board (white at the bottom, top-left first).")
    print(f"2. Squares are voted over the last {VOTE_WINDOW} readouts "
          f"({PREDICTION_INTERVAL}s apart).")
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

                    # One batched forward pass for the whole board
                    preds = predict_board(model, split_into_squares(warped))
                    last_prediction_time = current_time

                    for history, label in zip(histories, preds):
                        history.append(label)
                    voted_preds = majority_vote(histories)

                    digital_board_img = generate_digital_board(voted_preds)

                    # 3. GAME TRACKING: match the voted readout to a legal move
                    result = tracker.update(voted_preds)
                    if result["status"] == "move":
                        move_num = (len(tracker.moves) + 1) // 2
                        print(f"Move detected: {move_num}. {result['san']}")
                        print(f"  FEN: {tracker.fen}")
                        last_warned_placement = None
                    elif result["status"] == "new_game":
                        print("New game detected - tracker reset.")
                        last_warned_placement = None
                    elif result["status"] == "unrecognized":
                        # Warn once per distinct bad readout, not every cycle
                        if result["placement"] != last_warned_placement:
                            print("Readout doesn't match any legal move "
                                  "(noise, hand over board, or missed moves) - ignored.")
                            last_warned_placement = result["placement"]

                except Exception as e:
                    print(f"Prediction Error: {e}")

        # 4. Show Windows
        cv2.imshow("Live Feed", display_frame)
        cv2.imshow("Digital Board", digital_board_img)

        # 5. Input Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            points.clear()
            for history in histories:
                history.clear()
            print("Points reset.")
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print and save the final game record
    if tracker.moves:
        print("\nGame moves:", " ".join(tracker.moves))
        print(f"Final FEN: {tracker.fen}")
        path = tracker.save_pgn()
        print(f"Game saved to {path} - import it on lichess.org/paste for analysis.")


if __name__ == "__main__":
    main()
