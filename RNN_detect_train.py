import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# 器械类别定义
INSTRUMENT_CLASSES = {
    "剪刀": 0,
    "马里兰双极钳": 1,
    "单钳": 2,
    "针线夹": 3,
    "塑料夹": 4,
    "其他": 5,
    "塑料钳": 6  
}

# 数据集定义
class InstrumentDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for vid_folder in os.listdir(base_dir):
            vid_path = os.path.join(base_dir, vid_folder)
            if os.path.isdir(vid_path):
                left_frames_path = os.path.join(vid_path, 'left_frames')
                label_path = os.path.join(vid_path, 'label')

                for frame_name in os.listdir(left_frames_path):
                    frame_path = os.path.join(left_frames_path, frame_name)
                    if frame_path.endswith('.png') or frame_path.endswith('.jpg'):
                        label_file_name = frame_name.replace('.png', '.json').replace('.jpg', '.json')
                        label_file_path = os.path.join(label_path, label_file_name)

                        if not os.path.exists(label_file_path):
                            continue

                        with open(label_file_path, 'r', encoding='utf-8') as f:
                            label_data = json.load(f)

                        instrument_count = sum(
                            1 for shape in label_data.get('shapes', []) 
                            if shape['label'] in INSTRUMENT_CLASSES
                        )
                        instrument_count = min(instrument_count, 6)
                        self.labels.append(instrument_count)
                        self.image_paths.append(frame_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# 图像预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((1024, 1024)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

# 加载数据集
dataset = InstrumentDataset(base_dir=r'D:\yolov11\miccai', transform=transform)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 7)
    return model.to(device)

def train_model(model, train_loader, val_loader, epochs=100, lr=0.0001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    no_improvement_epochs = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        all_labels, all_predictions = [], []

        print(f"Epoch {epoch + 1}/{epochs}")
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        train_accuracy = correct_train / total_train
        avg_train_loss = running_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        print("Training Classification Report:")
        print(classification_report(all_labels, all_predictions, digits=4))

        val_loss, val_acc, val_report, _ = evaluate_model(model, val_loader)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        print("Validation Classification Report:")
        print(val_report)

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_epochs = 0
            torch.save(model.state_dict(), 'model_fold_best.pth')
            print(f"Saved best model at epoch {epoch + 1}")
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= 5:
            print("Early stopping.")
            break

def evaluate_model(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct, total = 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = running_loss / len(loader)
    acc = correct / total
    report = classification_report(all_labels, all_preds, digits=4)
    confusion_mat = confusion_matrix(all_labels, all_preds)
    return avg_loss, acc, report, confusion_mat

def test_model(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = correct / total
    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy

def main():
    # 划分数据集
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    test_size = int(0.2 * dataset_size)

    # 保留测试集
    _, remaining_indices = train_test_split(indices, test_size=test_size, shuffle=True)
    test_subset = Subset(dataset, test_size + remaining_indices[-test_size:])
    test_loader = DataLoader(test_subset, batch_size=8, shuffle=False)

    # 交叉验证
    kf = KFold(n_splits=2, shuffle=True)
    test_losses, test_accuracies = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\nFold {fold + 1}")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

        model = create_model()
        train_model(model, train_loader, val_loader)

        print("Evaluating on Test Set...")
        model.load_state_dict(torch.load("model_fold_best.pth"))
        test_loss, test_acc = test_model(model, test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    print(f"\nAverage Test Loss: {np.mean(test_losses):.4f}, Accuracy: {np.mean(test_accuracies):.4f}")

if __name__ == '__main__':
    main()
