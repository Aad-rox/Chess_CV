"""Shared board graphics and geometry helpers used by main.py and live_main.py."""

import os

import cv2
import numpy as np

ASSET_DIR = "assets"
BOARD_SIZE_PX = 600  # Size of the digital display window
WARP_SIZE_PX = 1000  # Warped board resolution (matches training data scale)

# Cache for loaded piece images, keyed by square size
_piece_images = {}


def load_piece_images(sq_size):
    """Loads piece PNGs from the assets folder, resized to sq_size."""
    loaded = {}
    labels = ["wk", "wq", "wr", "wb", "wn", "wp", "bk", "bq", "br", "bb", "bn", "bp"]

    for label in labels:
        path = os.path.join(ASSET_DIR, f"{label}.png")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # Load with alpha
            if img is not None:
                loaded[label] = cv2.resize(img, (sq_size, sq_size))
    return loaded


def overlay_transparent(background, overlay, x, y):
    """Overlays a PNG with an alpha channel onto a background image."""
    bg_h, bg_w, _ = background.shape
    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > bg_w or y + h > bg_h:
        return background

    if overlay.shape[2] < 4:
        # No alpha channel, just copy
        background[y:y + h, x:x + w] = overlay
        return background

    alpha_mask = overlay[:, :, 3] / 255.0
    overlay_img = overlay[:, :, :3]
    roi = background[y:y + h, x:x + w]

    for c in range(3):
        roi[:, :, c] = (alpha_mask * overlay_img[:, :, c] +
                        (1.0 - alpha_mask) * roi[:, :, c])
    background[y:y + h, x:x + w] = roi
    return background


def generate_digital_board(predictions, board_size_px=BOARD_SIZE_PX):
    """Draws a graphical board from a list of 64 square labels."""
    board_img = np.zeros((board_size_px, board_size_px, 3), dtype=np.uint8)
    sq_size = board_size_px // 8

    global _piece_images
    if not _piece_images or _piece_images.get("size") != sq_size:
        _piece_images = load_piece_images(sq_size)
        _piece_images["size"] = sq_size

    color_light = (240, 217, 181)  # Beige
    color_dark = (181, 136, 99)  # Brown

    for r in range(8):
        for c in range(8):
            y, x = r * sq_size, c * sq_size
            color = color_light if (r + c) % 2 == 0 else color_dark
            cv2.rectangle(board_img, (x, y), (x + sq_size, y + sq_size), color, -1)

            label = predictions[r * 8 + c]
            if label == "empty":
                continue

            if label in _piece_images:
                overlay_transparent(board_img, _piece_images[label], x, y)
            else:
                # Fallback: circle + text if no asset image found
                center = (x + sq_size // 2, y + sq_size // 2)
                fill = (0, 0, 0) if label.startswith('b') else (255, 255, 255)
                cv2.circle(board_img, center, sq_size // 3, fill, -1)
                cv2.circle(board_img, center, sq_size // 3, (0, 0, 0), 2)
                text_color = (255, 255, 255) if label.startswith('b') else (0, 0, 0)
                cv2.putText(board_img, label.upper(), (x + 10, y + sq_size - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    return board_img


def perspective_transform(image, src_pts, size=WARP_SIZE_PX):
    """Warps the four clicked corners to a square top-down view."""
    dst_pts = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, M, (size, size))


def split_into_squares(warped_image):
    """Splits a warped board image into 64 square crops, row-major from the top-left."""
    h, w = warped_image.shape[:2]
    sq_h, sq_w = h // 8, w // 8
    squares = []
    for r in range(8):
        for c in range(8):
            squares.append(warped_image[r * sq_h:(r + 1) * sq_h, c * sq_w:(c + 1) * sq_w])
    return squares
