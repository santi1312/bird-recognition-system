import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm

NUM_EPOCHS  = 10
BATCH_SIZE  = 32
LR          = 0.001
SAVE_PATH   = os.path.join("ml", "image", "bird_model.pt")
NAMES_PATH  = os.path.join("ml", "image", "class_names.json")
TRAIN_DIR   = os.path.join("ml", "image", "dataset", "train")
VAL_DIR     = os.path.join("ml", "image", "dataset", "valid")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device: " + str(DEVICE))
print("GPU: " + torch.cuda.get_device_name(0))
sys.stdout.flush()

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

print("Loading dataset from local folder...")
sys.stdout.flush()

train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset   = ImageFolder(VAL_DIR,   transform=val_transform)

NUM_CLASSES = len(train_dataset.classes)
class_names = train_dataset.classes

with open(NAMES_PATH, "w") as f:
    json.dump(class_names, f)

print("Classes: " + str(NUM_CLASSES))
print("Train: "   + str(len(train_dataset)) + " | Val: " + str(len(val_dataset)))
sys.stdout.flush()

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("Loading ResNet50...")
sys.stdout.flush()
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

for param in list(model.parameters())[:-20]:
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=3, gamma=0.5
)

best_val_acc = 0.0
print("Starting Training...")
sys.stdout.flush()

for epoch in range(NUM_EPOCHS):
    model.train()
    train_correct = 0
    train_total   = 0

    loop = tqdm(
        train_loader,
        desc="Epoch " + str(epoch+1) + "/" + str(NUM_EPOCHS) + " [Train]",
        file=sys.stdout
    )
    for images, labels in loop:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total   += labels.size(0)
        loop.set_postfix(loss=loss.item())

    train_acc = train_correct / train_total * 100 if train_total > 0 else 0

    model.eval()
    val_correct = 0
    val_total   = 0

    with torch.no_grad():
        for images, labels in tqdm(
            val_loader,
            desc="Epoch " + str(epoch+1) + "/" + str(NUM_EPOCHS) + " [Val]",
            file=sys.stdout
        ):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total   += labels.size(0)

    val_acc = val_correct / val_total * 100 if val_total > 0 else 0
    scheduler.step()

    print("Epoch " + str(epoch+1) + "/" + str(NUM_EPOCHS))
    print("  Train Acc: " + str(round(train_acc, 2)) + "%")
    print("  Val Acc:   " + str(round(val_acc, 2)) + "%")
    sys.stdout.flush()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print("  Best model saved! Val Acc: " + str(round(val_acc, 2)) + "%")
        sys.stdout.flush()

print("Training Complete!")
print("Best Val Accuracy: " + str(round(best_val_acc, 2)) + "%")
print("Model saved to: " + SAVE_PATH)