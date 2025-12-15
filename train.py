import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Import your model definition and classes
# Ensure square_classifier.py is inside the 'models' folder
from models.square_classifier import CLASSES
from torchvision.models import resnet18


def train():
    # --- CONFIGURATION ---
    DATA_DIR = "dataset_v2"
    SAVE_PATH = "models/weights.pth"
    BATCH_SIZE = 16
    EPOCHS = 24  # Increased to give time to learn variations
    LEARNING_RATE = 0.001
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Mac Silicon Support
    if torch.backends.mps.is_available():
        DEVICE = "mps"

    print(f"Training on {DEVICE}...")

    # --- THE MAGIC SAUCE: AUGMENTATION ---
    train_transform = transforms.Compose([
        # Note: If you re-ran the collector at 1000px, change this to (128, 128)
        # If you are still using the old data, keep it (64, 64)
        transforms.Resize((64, 64)),

        # 1. Rotation: Top-down pieces are circles.
        transforms.RandomRotation(degrees=25),

        # 2. Lighting: Simulates shadows or brighter webcam days.
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),

        # 3. Flips: A pawn is a pawn, whether on the left or right.
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),

        transforms.ToTensor(),
    ])

    # Load data
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Directory '{DATA_DIR}' not found.")
        return

    train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)

    # --- SAFETY CHECK ---
    detected_classes = train_dataset.classes
    if detected_classes != sorted(CLASSES):
        print("\n!!! WARNING !!!")
        print(f"Folder classes: {detected_classes}")
        print(f"Model classes:  {sorted(CLASSES)}")
        print("Classes do not match exactly. Double check your folder names.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- MODEL SETUP ---
    model = resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, len(detected_classes))
    model.to(DEVICE)

    # --- TRAINING LOOP ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    print("Starting training...")

    # TRACKING BEST ACCURACY
    best_acc = 0.0

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

        # Calculate epoch stats
        epoch_loss = running_loss / len(train_loader)
        acc = 100 * correct / total

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {acc:.2f}%")

        # --- SAVE ONLY IF BETTER ---
        if acc > best_acc:
            best_acc = acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  --> New Best Model! Saved weights ({acc:.2f}%)")

    print(f"\nDONE! Best accuracy achieved was: {best_acc:.2f}%")
    print(f"Weights saved to: {SAVE_PATH}")


if __name__ == "__main__":
    train()