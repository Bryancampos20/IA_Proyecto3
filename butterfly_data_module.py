import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl

class ButterflyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=64, unlabeled_ratio=0.0, labeled_ratio=1.0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.unlabeled_ratio = unlabeled_ratio
        self.labeled_ratio = labeled_ratio
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def setup(self, stage=None):
        # Cargar todo el conjunto de datos
        full_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.transform)
        labeled_size = int(len(full_dataset) * self.labeled_ratio)
        unlabeled_size = len(full_dataset) - labeled_size

        # Dividir el conjunto de datos en etiquetado y no etiquetado
        self.labeled_dataset, _ = random_split(full_dataset, [labeled_size, unlabeled_size])

        # Usa solo el conjunto etiquetado para entrenamiento
        if stage == 'fit' or stage is None:
            self.train_dataset = self.labeled_dataset
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'test'), transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)
