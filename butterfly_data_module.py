import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl

class ButterflyDataModule(pl.LightningDataModule):
    """
    DataModule para manejar el conjunto de datos de mariposas.
    
    Este módulo utiliza PyTorch Lightning para gestionar la carga y preparación de datos.
    Admite la división de los datos en conjuntos etiquetados y no etiquetados para entrenamientos semi-supervisados.
    """

    def __init__(self, data_dir, batch_size=64, unlabeled_ratio=0.0, labeled_ratio=1.0):
        """
        Inicializa el módulo de datos.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.unlabeled_ratio = unlabeled_ratio
        self.labeled_ratio = labeled_ratio
        # Transformaciones aplicadas a las imágenes
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Redimensionar las imágenes a 128x128 píxeles
            transforms.ToTensor(),         # Convertir las imágenes a tensores
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizar con media y desviación estándar
        ])

    def setup(self, stage=None):
        """
        Configura los conjuntos de datos para las diferentes etapas: entrenamiento y prueba.
        """
        # Cargar el conjunto de datos completo desde la carpeta 'train'
        full_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.transform)
        
        # Determinar la cantidad de datos etiquetados y no etiquetados
        labeled_size = int(len(full_dataset) * self.labeled_ratio)
        unlabeled_size = len(full_dataset) - labeled_size

        # Dividir el conjunto completo en etiquetado y no etiquetado
        self.labeled_dataset, _ = random_split(full_dataset, [labeled_size, unlabeled_size])

        # Asignar conjuntos de datos según la etapa
        if stage == 'fit' or stage is None:
            self.train_dataset = self.labeled_dataset  # Usar solo los datos etiquetados para entrenamiento
        if stage == 'test' or stage is None:
            # Cargar el conjunto de prueba desde la carpeta 'test'
            self.test_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'test'), transform=self.transform)

    def train_dataloader(self):
        """
        Retorna el DataLoader para el conjunto de entrenamiento.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def test_dataloader(self):
        """
        Retorna el DataLoader para el conjunto de prueba.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)
