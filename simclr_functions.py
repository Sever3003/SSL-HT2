# simclr_functions.py
import logging
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt

from torchvision.transforms.functional import to_pil_image
from PIL import Image
import torch

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SimCLR Model ---
class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        self.enc = base_encoder(pretrained=False)
        self.feature_dim = self.enc.fc.in_features

        self.enc.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc.maxpool = nn.Identity()
        self.enc.fc = nn.Identity()

        self.projector = nn.Linear(self.feature_dim, projection_dim)

    def forward(self, x):
        feature = self.enc(x)
        projection = self.projector(feature)
        return feature, projection

# --- Dataset Pair Augmentation ---
class CIFAR10Pair(CIFAR10):
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs), target

# --- NT-Xent Loss ---
def nt_xent(x, t=0.5):
    x = F.normalize(x, dim=1)
    similarity = x @ x.T / t
    mask = torch.eye(similarity.size(0), device=similarity.device).bool()
    similarity.masked_fill_(mask, float('-inf'))

    batch_size = x.size(0)
    targets = torch.arange(batch_size, device=x.device)
    targets = (targets + 1 - 2 * (targets % 2)).long()

    return F.cross_entropy(similarity, targets)

# --- Cosine Learning Rate Scheduler ---
def get_lr(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

# --- Training Functions ---
def train_simclr(model, train_loader, optimizer, scheduler, device, epochs):
    train_losses = []
    model.train()
    for epoch in range(epochs):
        total_loss, total_samples = 0.0, 0
        for x, _ in tqdm(train_loader):
            x = x.view(-1, *x.shape[2:]).to(device)
            optimizer.zero_grad()
            _, projections = model(x)
            loss = nt_xent(projections)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        avg_loss = total_loss / total_samples
        train_losses.append(avg_loss)
        logger.info(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

    return train_losses

class LinModel(nn.Module):
    def __init__(self, encoder, feature_dim, n_classes=10):
        super().__init__()
        self.enc = encoder
        self.lin = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        return self.lin(self.enc(x))

def train_linear_probe(model, train_loader, test_loader, optimizer, scheduler, device, epochs):
    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        correct_train, total_train = 0, 0
        total_train_loss = 0
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item() * x.size(0)
            correct_train += (logits.argmax(dim=1) == y).sum().item()
            total_train += y.size(0)

        train_losses.append(total_train_loss / total_train)
        train_acc = correct_train / total_train
        train_accuracies.append(train_acc)

        model.eval()
        correct_test, total_test = 0, 0
        total_test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                total_test_loss += loss.item() * x.size(0)
                correct_test += (logits.argmax(dim=1) == y).sum().item()
                total_test += y.size(0)

        test_losses.append(total_test_loss / total_test)
        test_acc = correct_test / total_test
        test_accuracies.append(test_acc)

        logger.info(f"Epoch {epoch+1}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    return train_losses, test_losses, train_accuracies, test_accuracies


def visualize_original_and_augmented(transform, images, num_examples=5):
    plt.figure(figsize=(15, 8))
    for i in range(num_examples):
        # Преобразование numpy в PIL.Image
        original_img = Image.fromarray(images[i].squeeze().astype('uint8'))
        
        # Аугментация
        augmented_img_tensor = transform(original_img)
        augmented_img = augmented_img_tensor.permute(1, 2, 0).numpy()
        augmented_img = (augmented_img * 0.25 + 0.5).clip(0, 1)
        plt.subplot(2, num_examples, i + 1)
        plt.imshow(original_img, cmap="gray")
        plt.title("Original")
        plt.axis("off")
        
        plt.subplot(2, num_examples, i + 1 + num_examples)
        plt.imshow(augmented_img.squeeze(), cmap="gray")
        plt.title("Augmented")
        plt.axis("off")
    
    plt.suptitle("Original and Augmented Images")
    plt.show()

def visualize_original_and_augmented_set(transform, dataset, num_examples=5):

    plt.figure(figsize=(15, 6))
    for i in range(num_examples):
        idx = torch.randint(0, len(dataset), (1,)).item()
        original_img, _ = dataset[idx]
        
        original_img = Image.fromarray(dataset.data[idx])  
        

        augmented_img_tensor = transform(original_img)
        augmented_img = augmented_img_tensor.permute(1, 2, 0).numpy() 
        

        plt.subplot(2, num_examples, i + 1)
        plt.imshow(original_img)
        plt.title("Original")
        plt.axis("off")
        
     
        plt.subplot(2, num_examples, i + 1 + num_examples)
        plt.imshow((augmented_img * 0.5 + 0.5).clip(0, 1)) 
        plt.title("Augmented")
        plt.axis("off")
    
    plt.suptitle("Original and Augmented Images", fontsize=16)
    plt.show()

    import torch
from torch.utils.data import Dataset
from PIL import Image

class GenericPairDataset(Dataset):
    """
    Генерирует пары аугментированных изображений из любого датасета.
    """
    def __init__(self, dataset, transform=None):
        """
        Args:
            dataset: Базовый датасет (например, ChestMNIST или PneumoniaMNIST).
            transform: Трансформации, которые будут применяться к изображениям.
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]

        # Если изображение в формате numpy, преобразуем в PIL.Image
        if isinstance(img, (torch.Tensor, np.ndarray)):
            img = Image.fromarray(img.squeeze().astype('uint8'))

        # Применяем аугментации дважды для создания пары
        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)
        else:
            img1 = img2 = img

        return torch.stack([img1, img2]), target


def plot_results(train_loss, val_loss, train_acc, val_acc):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # График потерь
    axs[0].plot(train_loss, label="Train Loss", marker='o')
    axs[0].plot(val_loss, label="Validation Loss", marker='o')
    axs[0].set_title("Losses")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    # График точности
    axs[1].plot(train_acc, label="Train Accuracy", marker='o')
    axs[1].plot(val_acc, label="Validation Accuracy", marker='o')
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].grid(True)

    plt.show()

