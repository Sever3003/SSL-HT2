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
from torch.utils.data import Dataset

from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet18, ResNet18_Weights

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SimCLR Model ---
class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128, pretrained=False):
        super().__init__()

        if pretrained:
            self.enc = base_encoder(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.enc = base_encoder(pretrained=False)

        self.feature_dim = self.enc.fc.in_features

        self.enc.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=7, padding=7, bias=False)
        # self.enc.maxpool = nn.Identity()
        self.enc.fc = nn.Identity()

        self.projector = nn.Linear(self.feature_dim, projection_dim)

    def forward(self, x):
        feature = self.enc(x)
        projection = self.projector(feature)
        return feature, projection
    
# --- Dataset Pair Augmentation ---
class GenericPairDataset(Dataset):
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

        if isinstance(img, (torch.Tensor, np.ndarray)):
            img = Image.fromarray(img.squeeze().astype('uint8'))

        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)
        else:
            img1 = img2 = img

        return torch.stack([img1, img2]), target
    

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



from sklearn.metrics import f1_score, precision_score, recall_score
import torch.nn.functional as F

# --- Multi-label Linear Model ---
class MultiLabelLinModel(nn.Module):
    def __init__(self, encoder, feature_dim, n_classes):
        super().__init__()
        self.enc = encoder
        self.lin = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        return self.lin(self.enc(x))

# --- Multi-label Training Function ---
def train_multilabel_probe(model, train_loader, test_loader, optimizer, scheduler, device, epochs, class_weights=None):
    train_losses, test_losses = [], []
    train_f1_micro, test_f1_micro = [], []
    train_f1_macro, test_f1_macro = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        all_train_preds, all_train_targets = [], []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, weight=class_weights)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item() * x.size(0)
            preds = torch.sigmoid(logits).detach().cpu().numpy()
            all_train_preds.append(preds)
            all_train_targets.append(y.cpu().numpy())

        train_losses.append(total_train_loss / len(train_loader.dataset))

        # Metrics for training
        train_preds = np.concatenate(all_train_preds)
        train_targets = np.concatenate(all_train_targets)
        train_f1_micro.append(f1_score(train_targets, train_preds > 0.5, average='micro'))
        train_f1_macro.append(f1_score(train_targets, train_preds > 0.5, average='macro'))

        # Validation phase
        model.eval()
        total_test_loss = 0
        all_test_preds, all_test_targets = [], []

        with torch.no_grad():
            for x, y in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} - Testing"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y, weight=class_weights)
                total_test_loss += loss.item() * x.size(0)
                preds = torch.sigmoid(logits).cpu().numpy()
                all_test_preds.append(preds)
                all_test_targets.append(y.cpu().numpy())

        test_losses.append(total_test_loss / len(test_loader.dataset))

        # Metrics for testing
        test_preds = np.concatenate(all_test_preds)
        test_targets = np.concatenate(all_test_targets)
        test_f1_micro.append(f1_score(test_targets, test_preds > 0.5, average='micro'))
        test_f1_macro.append(f1_score(test_targets, test_preds > 0.5, average='macro'))

        logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss={train_losses[-1]:.4f}, Test Loss={test_losses[-1]:.4f}, "
                    f"Train Micro F1={train_f1_micro[-1]:.4f}, Test Micro F1={test_f1_micro[-1]:.4f}, "
                    f"Train Macro F1={train_f1_macro[-1]:.4f}, Test Macro F1={test_f1_macro[-1]:.4f}")

    return train_losses, test_losses, train_f1_micro, test_f1_micro, train_f1_macro, test_f1_macro






def visualize(transform, dataset, num_examples=5):
    """
    Визуализирует оригинальные и аугментированные изображения из датасета.

    Args:
        transform: Трансформация, применяемая к изображениям.
        dataset: Датасет, содержащий изображения.
        num_examples: Количество примеров для отображения.
    """
    plt.figure(figsize=(15, 6))
    for i in range(num_examples):
        # Выбираем случайное изображение из датасета
        idx = torch.randint(0, len(dataset), (1,)).item()
        original_img, _ = dataset[idx]

        # Преобразуем оригинал из Tensor в PIL.Image, если необходимо
        if isinstance(original_img, torch.Tensor):
            original_img = to_pil_image(original_img)

        # Применяем аугментацию
        augmented_img = transform(original_img)

        # Отображаем оригинальное изображение
        plt.subplot(2, num_examples, i + 1)
        plt.imshow(original_img, cmap='gray')
        plt.title("Original")
        plt.axis("off")

        # Отображаем аугментированное изображение
        plt.subplot(2, num_examples, i + 1 + num_examples)
        augmented_img = to_pil_image(augmented_img)
        plt.imshow(augmented_img, cmap='gray')
        plt.title("Augmented")
        plt.axis("off")

    plt.suptitle("Original and Augmented Images")
    plt.show()

def plot_results(train_loss, val_loss, train_f1, val_f1, train_recall, val_recall, save_dir):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # График потерь
    axs[0].plot(train_loss, label="Train Loss", marker='o')
    axs[0].plot(val_loss, label="Validation Loss", marker='o')
    axs[0].set_title("Losses")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    # График F1-Score
    axs[1].plot(train_f1, label="Train F1-Score", marker='o')
    axs[1].plot(val_f1, label="Validation F1-Score", marker='o')
    axs[1].set_title("F1-Score")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("F1-Score")
    axs[1].legend()
    axs[1].grid(True)

    # График Recall
    axs[2].plot(train_recall, label="Train Recall", marker='o')
    axs[2].plot(val_recall, label="Validation Recall", marker='o')
    axs[2].set_title("Recall")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("Recall")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_dir)
    plt.show()
    # Сохраняем график