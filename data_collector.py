import cv2
import numpy as np
import os
import glob

# --- CONFIGURATION ---
INPUT_FOLDER = "raw_photos"  # Put your 30-50 raw board photos here
DATASET_DIR = "dataset_v2"  # Where the cropped squares will be saved

# THE "PACKED" LAYOUT (Must match your physical board arrangement)
GRID_LAYOUT = [
    # Row 0 (Top)
    "wr", "wn", "wb", "wq", "wk", "wb", "wn", "wr",
    # Row 1
    "wp", "wp", "wp", "wp", "wp", "wp", "wp", "wp",
    # Row 2
    "bp", "bp", "bp", "bp", "bp", "bp", "bp", "bp",
    # Row 3
    "br", "bn", "bb", "bq", "bk", "bb", "bn", "br",
    # Rows 4-7 (Empty)
    "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty",
    "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty",
    "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty",
    "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty",
]

# Ensure output directories exist
for label in set(GRID_LAYOUT):
    os.makedirs(os.path.join(DATASET_DIR, label), exist_ok=True)

# Global variables
points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")


def perspective_transform(image, src_pts):
    dst_pts = np.array([[0, 0], [500, 0], [500, 500], [0, 500]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, M, (500, 500))


def split_and_save(warped_image, original_filename):
    height, width = warped_image.shape[:2]
    sq_h, sq_w = height // 8, width // 8

    base_name = os.path.splitext(os.path.basename(original_filename))[0]

    for r in range(8):
        for c in range(8):
            start_x = c * sq_w
            start_y = r * sq_h
            square_img = warped_image[start_y:start_y + sq_h, start_x:start_x + sq_w]

            idx = r * 8 + c
            label = GRID_LAYOUT[idx]

            # Save: dataset_v2/wk/photo1_0_4.png
            filename = f"{base_name}_{r}_{c}.png"
            save_path = os.path.join(DATASET_DIR, label, filename)
            cv2.imwrite(save_path, square_img)

    print(f"Saved squares from {base_name}!")


def main():
    # 1. Get list of all images
    # Supports jpg, png, jpeg. Add more if needed.
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

    image_files.sort()  # Process in order

    if not image_files:
        print(f"No images found in '{INPUT_FOLDER}'. Create the folder and add photos!")
        return

    print(f"Found {len(image_files)} images. Starting batch process...")

    cv2.namedWindow("Batch Collector")
    cv2.setMouseCallback("Batch Collector", mouse_callback)

    current_idx = 0

    while current_idx < len(image_files):
        img_path = image_files[current_idx]
        image = cv2.imread(img_path)

        if image is None:
            print(f"Could not load {img_path}, skipping...")
            current_idx += 1
            continue

        # Resize for easier clicking if image is huge (4k phone photos)
        # We will scale points back up later if needed, but keeping it simple:
        # Just display it scaled down, but process the original is tricky.
        # EASIER: Just resize the image to a manageable size (e.g. 1000px width)
        h, w = image.shape[:2]
        if w > 1200:
            scale = 1200 / w
            image = cv2.resize(image, (1200, int(h * scale)))

        # Reset points for new image
        global points
        points = []

        print(f"Processing [{current_idx + 1}/{len(image_files)}]: {os.path.basename(img_path)}")
        print("Click 4 corners (White Rook -> TopRight -> BottomRight -> BottomLeft)")

        while True:
            display_img = image.copy()

            for i, pt in enumerate(points):
                cv2.circle(display_img, pt, 5, (0, 255, 0), -1)
                cv2.putText(display_img, str(i + 1), (pt[0] + 10, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if len(points) == 4:
                cv2.polylines(display_img, [np.array(points)], True, (255, 0, 0), 2)

                # Show Preview
                try:
                    src_pts = np.array(points, dtype="float32")
                    warped = perspective_transform(image, src_pts)
                    cv2.imshow("Preview", warped)
                except:
                    pass

            cv2.imshow("Batch Collector", display_img)

            key = cv2.waitKey(1) & 0xFF

            # ACTION: Save and Next
            if key == ord('s') and len(points) == 4:
                src_pts = np.array(points, dtype="float32")
                warped = perspective_transform(image, src_pts)
                split_and_save(warped, img_path)

                # Move to next image automatically
                current_idx += 1
                cv2.destroyWindow("Preview")  # Close preview
                break  # Break inner loop to load next image

            # ACTION: Skip Image
            if key == ord('n'):
                print("Skipping image...")
                current_idx += 1
                break

            # ACTION: Reset Points
            if key == ord('r'):
                points = []
                print("Points reset.")

            # ACTION: Quit
            if key == ord('q'):
                print("Quitting.")
                return

    cv2.destroyAllWindows()
    print("All images processed!")


if __name__ == "__main__":
    main()