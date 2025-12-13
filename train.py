import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Import your existing model definition and classes
# Ensure square_classifier.py is inside the 'models' folder
from models.square_classifier import load_model, CLASSES
from torchvision.models import resnet18


def train():
    # --- CONFIGURATION ---
    DATA_DIR = "data"  # Folder containing 'empty', 'wp', 'bn', etc.
    SAVE_PATH = "models/weights.pth"
    BATCH_SIZE = 32
    EPOCHS = 5  # 5 epochs is usually enough for this small task
    LEARNING_RATE = 0.001
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {DEVICE}...")

    # --- DATA PREPARATION ---
    # These transforms must match square_classifier.py exactly
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Load data
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Directory '{DATA_DIR}' not found. Please create it.")
        return

    train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=data_transform)

    # Check classes
    print(f"Classes found: {train_dataset.classes}")
    if train_dataset.classes != sorted(CLASSES):
        print("\nWARNING: Your data folders do not match the model classes exactly!")
        print(f"Model expects: {sorted(CLASSES)}")
        print(f"Data found:  {train_dataset.classes}")
        print("Please check folder names (e.g., 'empty' vs 'Empty').\n")
        # We continue, but this might cause prediction errors later if names don't align.

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- MODEL SETUP ---
    # Initialize a fresh ResNet18
    model = resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.to(DEVICE)

    # --- TRAINING LOOP ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Loss: {running_loss / len(train_loader):.4f} | Acc: {100 * correct / total:.2f}%")

    # --- SAVE WEIGHTS ---
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nDONE! Weights saved to: {SAVE_PATH}")


if __name__ == "__main__":
    train()