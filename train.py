import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from collections import defaultdict
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
    EPOCHS = 24
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.2  # Hold out 20% of each class to measure real accuracy
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Mac Silicon Support
    if torch.backends.mps.is_available():
        DEVICE = "mps"

    print(f"Training on {DEVICE}...")

    # --- THE MAGIC SAUCE: AUGMENTATION ---
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),

        # 1. Rotation + translation + scale: top-down pieces have no canonical
        # orientation, and live grid crops are never perfectly centered.
        transforms.RandomAffine(degrees=180, translate=(0.1, 0.1), scale=(0.85, 1.1)),

        # 2. Perspective: the camera is never perfectly overhead, so edge
        # squares get a slight trapezoid distortion.
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),

        # 3. Lighting: simulates shadows or brighter webcam days.
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),

        # 4. Flips: a pawn is a pawn, whether on the left or right.
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),

        transforms.ToTensor(),
    ])

    # Validation images get NO augmentation - we want to measure the model
    # on clean images, the same way it sees them at inference time.
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Load data
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Directory '{DATA_DIR}' not found.")
        return

    # Two views of the same folder: augmented for training, clean for validation
    train_folder = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
    val_folder = datasets.ImageFolder(root=DATA_DIR, transform=val_transform)

    # --- SAFETY CHECK ---
    detected_classes = train_folder.classes
    if detected_classes != sorted(CLASSES):
        print("\n!!! WARNING !!!")
        print(f"Folder classes: {detected_classes}")
        print(f"Model classes:  {sorted(CLASSES)}")
        print("Classes do not match exactly. Double check your folder names.")

    # --- STRATIFIED TRAIN/VAL SPLIT ---
    # Split each class separately so rare classes (kings/queens have only ~40
    # images) are guaranteed to appear in both sets.
    indices_by_class = defaultdict(list)
    for idx, target in enumerate(train_folder.targets):
        indices_by_class[target].append(idx)

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = [], []
    for target, indices in indices_by_class.items():
        perm = torch.randperm(len(indices), generator=generator).tolist()
        n_val = max(1, int(len(indices) * VAL_SPLIT))
        val_indices += [indices[i] for i in perm[:n_val]]
        train_indices += [indices[i] for i in perm[n_val:]]

    train_dataset = Subset(train_folder, train_indices)
    val_dataset = Subset(val_folder, val_indices)
    print(f"Split: {len(train_dataset)} train / {len(val_dataset)} val images")

    # --- WEIGHTED SAMPLING ---
    # 'empty' is half the dataset; sample rare classes more often so the model
    # sees every class equally during training.
    train_targets = torch.tensor([train_folder.targets[i] for i in train_indices])
    class_counts = torch.bincount(train_targets, minlength=len(detected_classes))
    sample_weights = 1.0 / class_counts[train_targets].float()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_indices), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # --- MODEL SETUP ---
    model = resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, len(detected_classes))
    model.to(DEVICE)

    # --- TRAINING LOOP ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")

    # TRACKING BEST VALIDATION ACCURACY
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
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

        epoch_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # --- VALIDATION ---
        model.eval()
        val_correct = 0
        val_total = 0
        per_class_correct = torch.zeros(len(detected_classes))
        per_class_total = torch.zeros(len(detected_classes))
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                predicted = model(images).argmax(dim=1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                for c in range(len(detected_classes)):
                    mask = labels == c
                    per_class_total[c] += mask.sum().item()
                    per_class_correct[c] += (predicted[mask] == c).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {epoch_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # --- SAVE ONLY IF BETTER ON VALIDATION ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  --> New Best Model! Saved weights ({val_acc:.2f}% val)")
            best_per_class = (per_class_correct, per_class_total)

    print(f"\nDONE! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Weights saved to: {SAVE_PATH}")

    print("\nPer-class validation accuracy (at best epoch):")
    correct_t, total_t = best_per_class
    for c, name in enumerate(detected_classes):
        if total_t[c] > 0:
            print(f"  {name}: {int(correct_t[c])}/{int(total_t[c])} = "
                  f"{100 * correct_t[c] / total_t[c]:.1f}%")


if __name__ == "__main__":
    train()
