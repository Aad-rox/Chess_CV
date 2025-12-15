import cv2
import numpy as np
import os
import requests
from models.square_classifier import load_model, predict_square

# --- CONFIGURATION ---
ASSET_DIR = "assets"
BOARD_SIZE_PX = 600  # Size of the digital display window

# URLs for standard chess piece images (Wikimedia Commons)
PIECE_URLS = {
    "wk": "https://upload.wikimedia.org/wikipedia/commons/4/42/Chess_klt45.svg",
    "wq": "https://upload.wikimedia.org/wikipedia/commons/1/15/Chess_qlt45.svg",
    "wr": "https://upload.wikimedia.org/wikipedia/commons/7/72/Chess_rlt45.svg",
    "wb": "https://upload.wikimedia.org/wikipedia/commons/b/b1/Chess_blt45.svg",
    "wn": "https://upload.wikimedia.org/wikipedia/commons/7/70/Chess_nlt45.svg",
    "wp": "https://upload.wikimedia.org/wikipedia/commons/4/45/Chess_plt45.svg",
    "bk": "https://upload.wikimedia.org/wikipedia/commons/f/f0/Chess_kdt45.svg",
    "bq": "https://upload.wikimedia.org/wikipedia/commons/4/47/Chess_qdt45.svg",
    "br": "https://upload.wikimedia.org/wikipedia/commons/f/ff/Chess_rdt45.svg",
    "bb": "https://upload.wikimedia.org/wikipedia/commons/9/98/Chess_bdt45.svg",
    "bn": "https://upload.wikimedia.org/wikipedia/commons/e/ef/Chess_ndt45.svg",
    "bp": "https://upload.wikimedia.org/wikipedia/commons/c/c7/Chess_pdt45.svg",
}


# Note: We will download PNG versions of these by changing the URL logic slightly
# or just using a direct PNG source. To keep it robust, let's use a simpler PNG source
# if you don't have SVG support.
# ACTUALLY: Let's use a simple placeholder logic if images aren't present,
# or download PNGs directly. Wikimedia requires user-agent headers sometimes.
# Let's use a direct PNG mirror for simplicity.

def download_assets():
    """Downloads chess piece PNGs if they don't exist."""
    if not os.path.exists(ASSET_DIR):
        os.makedirs(ASSET_DIR)

    # Using a reliable GitHub mirror for PNG chess pieces
    base_url = "https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett/"

    # Map our labels to filename format
    # Our labels: "wk", "bp" -> Lichess filenames: "wK.svg", "bP.svg" (We need PNGs though)
    # Let's use a different source that provides PNGs directly to avoid conversion issues.
    # Source: PyChess assets or similar.

    print("Checking assets...")
    # Map our internal labels to standard filenames
    pieces = {
        "wk": "wK.png", "wq": "wQ.png", "wr": "wR.png", "wb": "wB.png", "wn": "wN.png", "wp": "wP.png",
        "bk": "bK.png", "bq": "bQ.png", "br": "bR.png", "bb": "bB.png", "bn": "bN.png", "bp": "bP.png"
    }

    # We will assume you have these images. IF NOT, I will create simple colored text placeholders
    # so the code doesn't crash.
    # (Writing a robust downloader in this snippet is complex because of URL variations).
    return pieces


# Load the model
model = load_model("models/weights.pth")

# Global variables
points = []
piece_images = {}  # Cache for loaded images


def load_piece_images(sq_size):
    """
    Loads images from 'assets/' folder.
    If not found, we will handle it gracefully in drawing.
    """
    loaded = {}
    labels = ["wk", "wq", "wr", "wb", "wn", "wp", "bk", "bq", "br", "bb", "bn", "bp"]

    # Create dummy assets if files are missing (Blue/Red circles with text)
    # This ensures code RUNS even without downloading anything.
    for label in labels:
        # Try to load real image if you have them (e.g. assets/wk.png)
        path = os.path.join(ASSET_DIR, f"{label}.png")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # Load with Alpha
            if img is not None:
                loaded[label] = cv2.resize(img, (sq_size, sq_size))
    return loaded


def overlay_transparent(background, overlay, x, y):
    """
    Overlays a PNG with transparency onto a background image.
    """
    bg_h, bg_w, _ = background.shape
    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > bg_w or y + h > bg_h:
        return background

    if overlay.shape[2] < 4:
        # No alpha channel, just copy
        background[y:y + h, x:x + w] = overlay
        return background

    # Separate alpha and color
    alpha_mask = overlay[:, :, 3] / 255.0
    overlay_img = overlay[:, :, :3]

    # Region of interest
    roi = background[y:y + h, x:x + w]

    # Blend
    for c in range(0, 3):
        roi[:, :, c] = (alpha_mask * overlay_img[:, :, c] +
                        (1.0 - alpha_mask) * roi[:, :, c])

    background[y:y + h, x:x + w] = roi
    return background


def generate_digital_board(predictions):
    # 1. Create blank board
    board_img = np.zeros((BOARD_SIZE_PX, BOARD_SIZE_PX, 3), dtype=np.uint8)
    sq_size = BOARD_SIZE_PX // 8

    # Load assets if not already loaded (or re-load if size changed)
    global piece_images
    if not piece_images or piece_images.get("size") != sq_size:
        piece_images = load_piece_images(sq_size)
        piece_images["size"] = sq_size

    # Colors (light, dark)
    color_light = (240, 217, 181)  # Beige
    color_dark = (181, 136, 99)  # Brown

    for r in range(8):
        for c in range(8):
            # 2. Draw Background Square
            y = r * sq_size
            x = c * sq_size
            color = color_light if (r + c) % 2 == 0 else color_dark
            cv2.rectangle(board_img, (x, y), (x + sq_size, y + sq_size), color, -1)

            # 3. Draw Piece
            label = predictions[r * 8 + c]
            if label != "empty":
                if label in piece_images:
                    # Use loaded PNG
                    overlay_transparent(board_img, piece_images[label], x, y)
                else:
                    # FALLBACK: Draw circle + Text if no image found
                    center = (x + sq_size // 2, y + sq_size // 2)
                    color = (0, 0, 0) if label.startswith('b') else (255, 255, 255)
                    cv2.circle(board_img, center, sq_size // 3, color, -1)
                    cv2.circle(board_img, center, sq_size // 3, (0, 0, 0), 2)  # Outline
                    # Draw text label
                    text_color = (255, 255, 255) if label.startswith('b') else (0, 0, 0)
                    cv2.putText(board_img, label.upper(), (x + 10, y + sq_size - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    return board_img


# ... (Include your detect_chessboard, perspective_transform, split_squares from before) ...
# I will repeat the MAIN loop here for clarity

def main():
    # 1. Download/Setup Assets Instructions
    if not os.path.exists(ASSET_DIR):
        os.makedirs(ASSET_DIR)
        print(f"WARNING: '{ASSET_DIR}' folder is empty.")
        print(
            "To see real icons, download chess piece PNGs (filenames: wk.png, bp.png, etc.) into the 'assets' folder.")
        print("For now, the code will use colored circles as placeholders.")

    image = cv2.imread("imgs/test2.JPG")
    if image is None:
        print("Error loading image.")
        return

    # Standard Mouse Callback Setup
    cv2.namedWindow("Chessboard")
    cv2.setMouseCallback("Chessboard", mouse_callback)

    while True:
        img_copy = image.copy()

        # Draw selection points
        for point in points:
            cv2.circle(img_copy, point, 5, (0, 255, 0), -1)
        if len(points) == 4:
            cv2.polylines(img_copy, [np.array(points)], isClosed=True, color=(255, 0, 0), thickness=2)

        cv2.imshow("Chessboard", img_copy)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('p') and len(points) == 4:
            src_pts = np.array(points, dtype="float32")
            warped_image = perspective_transform(image, src_pts)
            all_squares = split_chessboard_into_squares(warped_image)

            print("Predicting...")
            preds = []
            for sq in all_squares:
                preds.append(predict_square(model, sq))
            print("Done.")

            # --- GENERATE DIGITAL BOARD (MANUAL DRAWING) ---
            digital_board_img = generate_digital_board(preds)
            cv2.imshow("Digital State", digital_board_img)

        if key == ord('q'):
            break

        if key == ord('d'):
            print("Debug Mode: Showing standard starting position...")
            # Hardcoded list of labels for a standard chess starting position
            debug_preds = [
                "br", "bn", "bb", "bq", "bk", "bb", "bn", "br",
                "bp", "bp", "bp", "bp", "bp", "bp", "bp", "bp",
                "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty",
                "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty",
                "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty",
                "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty",
                "wp", "wp", "wp", "wp", "wp", "wp", "wp", "wp",
                "wr", "wn", "wb", "wq", "wk", "wb", "wn", "wr"
            ]

            # Generate the board using these fake labels
            digital_board_img = generate_digital_board(debug_preds)
            cv2.imshow("Digital State", digital_board_img)

    cv2.destroyAllWindows()


# Helper wrappers needed for the simplified main above
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Point selected: ({x}, {y})")


def perspective_transform(image, src_pts):
    dst_pts = np.array([[0, 0], [500, 0], [500, 500], [0, 500]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, M, (500, 500))


def split_chessboard_into_squares(warped_image, board_size=8):
    height, width = warped_image.shape[:2]
    squares = []
    sq_h, sq_w = height // board_size, width // board_size
    for r in range(board_size):
        for c in range(board_size):
            squares.append(warped_image[r * sq_h:(r + 1) * sq_h, c * sq_w:(c + 1) * sq_w])
    return squares


if __name__ == "__main__":
    main()