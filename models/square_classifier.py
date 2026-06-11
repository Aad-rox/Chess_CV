import cv2  # <--- Need this for color conversion
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18

# ALPHABETICAL ORDER (Standard for PyTorch ImageFolder)
CLASSES = [
    "bb", "bk", "bn", "bp", "bq", "br",
    "empty",
    "wb", "wk", "wn", "wp", "wq", "wr"
]

def load_model(weights_path):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


transform = T.Compose([
    T.ToPILImage(),
    T.Resize((64, 64)),
    T.ToTensor()
])


def get_device():
    """Best available device: Apple GPU (mps), NVIDIA (cuda), or cpu."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def predict_square(model, square_img):
    # 1. CRITICAL FIX: Convert OpenCV BGR to RGB
    # The model was trained on RGB images, but OpenCV gives BGR.
    rgb_square = cv2.cvtColor(square_img, cv2.COLOR_BGR2RGB)

    # 2. Transform the corrected image
    x = transform(rgb_square).unsqueeze(0)
    x = x.to(next(model.parameters()).device)

    with torch.no_grad():
        logits = model(x)
    return CLASSES[logits.argmax(dim=1).item()]


def predict_board(model, squares):
    """Classifies a list of square crops in a single batched forward pass.

    Much faster than calling predict_square per square: the whole board
    goes through the network at once instead of 64 separate passes.
    Returns a list of labels in the same order as the input.
    """
    batch = torch.stack([
        transform(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)) for sq in squares
    ])
    batch = batch.to(next(model.parameters()).device)

    with torch.no_grad():
        logits = model(batch)
    return [CLASSES[i] for i in logits.argmax(dim=1).tolist()]