import cv2
import numpy as np
import time
import torch
import os
from PIL import Image

# Import your existing modules
from models.square_classifier import load_model, predict_square, CLASSES

# --- CONFIGURATION ---
PREDICTION_INTERVAL = 2.0  # Updates board every 2 seconds
WEBCAM_ID = 0  # Change to 1 if using an external webcam
BOARD_SIZE_PX = 600  # Size of the digital display

# --- LOAD RESOURCES ---
# Load Model
model = load_model("models/weights.pth")
device = "cpu"
model.to(device)
model.eval()

# Load Assets (Simple Dictionary Cache)
ASSET_DIR = "assets"
piece_images = {}


def load_piece_images(sq_size):
    """Loads PNGs from assets folder."""
    loaded = {}
    # Internal labels mapped to filename
    labels = ["wk", "wq", "wr", "wb", "wn", "wp", "bk", "bq", "br", "bb", "bn", "bp"]

    for label in labels:
        path = os.path.join(ASSET_DIR, f"{label}.png")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                loaded[label] = cv2.resize(img, (sq_size, sq_size))
    return loaded


# --- GRAPHICS HELPERS ---
def overlay_transparent(background, overlay, x, y):
    """Overlays PNG with alpha channel."""
    bg_h, bg_w, _ = background.shape
    h, w = overlay.shape[0], overlay.shape[1]
    if x + w > bg_w or y + h > bg_h: return background

    # Check if alpha exists
    if overlay.shape[2] < 4:
        background[y:y + h, x:x + w] = overlay
        return background

    alpha_mask = overlay[:, :, 3] / 255.0
    overlay_img = overlay[:, :, :3]
    roi = background[y:y + h, x:x + w]

    for c in range(3):
        roi[:, :, c] = (alpha_mask * overlay_img[:, :, c] + (1.0 - alpha_mask) * roi[:, :, c])
    background[y:y + h, x:x + w] = roi
    return background


def generate_digital_board(predictions):
    """Draws the graphical board."""
    board_img = np.zeros((BOARD_SIZE_PX, BOARD_SIZE_PX, 3), dtype=np.uint8)
    sq_size = BOARD_SIZE_PX // 8

    global piece_images
    if not piece_images:
        piece_images = load_piece_images(sq_size)

    color_light = (240, 217, 181)
    color_dark = (181, 136, 99)

    for r in range(8):
        for c in range(8):
            y, x = r * sq_size, c * sq_size
            color = color_light if (r + c) % 2 == 0 else color_dark
            cv2.rectangle(board_img, (x, y), (x + sq_size, y + sq_size), color, -1)

            label = predictions[r * 8 + c]

            # Draw Piece
            if label != "empty" and label in piece_images:
                overlay_transparent(board_img, piece_images[label], x, y)
            elif label != "empty":
                # Fallback text
                cv2.putText(board_img, label, (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return board_img


def perspective_transform(image, src_pts):
    # Use 1000px to match your high-res training data
    dst_pts = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, M, (1000, 1000))


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

    print("--- LIVE MODE ---")
    print("1. Click 4 corners of the board.")
    print("2. The digital board updates every 2.0 seconds.")
    print("3. Press 'r' to reset corners.")
    print("4. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. UI: Draw Circles/Lines
        display_frame = frame.copy()
        for i, pt in enumerate(points):
            cv2.circle(display_frame, pt, 5, (0, 255, 0), -1)
            cv2.putText(display_frame, str(i + 1), (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                        2)

        if len(points) == 4:
            cv2.polylines(display_frame, [np.array(points)], True, (255, 0, 0), 2)

            # 2. TIME CHECK: Is it time to predict?
            current_time = time.time()
            if current_time - last_prediction_time > PREDICTION_INTERVAL:

                try:
                    # A. Warp
                    src_pts = np.array(points, dtype="float32")
                    warped = perspective_transform(frame, src_pts)

                    # B. Split
                    squares = []
                    h, w = warped.shape[:2]
                    sq_h, sq_w = h // 8, w // 8
                    for r in range(8):
                        for c in range(8):
                            squares.append(warped[r * sq_h:(r + 1) * sq_h, c * sq_w:(c + 1) * sq_w])

                    # C. Predict (Simple Loop)
                    current_preds = []
                    for sq in squares:
                        # STANDARD CALL (No confidence logic)
                        lbl = predict_square(model, sq)
                        current_preds.append(lbl)

                    # D. Update Graphics
                    digital_board_img = generate_digital_board(current_preds)

                    last_prediction_time = current_time
                    print("Updated board.")

                except Exception as e:
                    print(f"Prediction Error: {e}")

        # 3. Show Windows
        cv2.imshow("Live Feed", display_frame)
        cv2.imshow("Digital Board", digital_board_img)

        # 4. Input Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            points.clear()
            print("Points reset.")
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()