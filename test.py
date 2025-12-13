import cv2
import sys
import os
# Import your specific loading and prediction functions
from models.square_classifier import load_model, predict_square


def test_single_image(image_path):
    # 1. Load the model
    print("Loading model...")
    model = load_model("models/weights.pth")

    # 2. Load the test image
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return

    # OpenCV loads images as BGR
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not read image. Is it a valid image file?")
        return

    # 3. Run Prediction
    # This calls your function which now handles BGR -> RGB conversion
    print(f"Testing image: {image_path}")
    label = predict_square(model, image)

    print("-" * 30)
    print(f"PREDICTION: {label}")
    print("-" * 30)


if __name__ == "__main__":
    # You can run this from command line: python test_single_image.py my_pawn.jpg
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        # REPLACE THIS with the path to one of your original training images
        test_path = "data/bk/resized_1691059574888.jpg"

    test_single_image(test_path)