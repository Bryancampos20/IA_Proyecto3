import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score

# Clasificador A: Modelo básico entrenado desde cero
class ClassifierA(pl.LightningModule):
    """
    Modelo clasificador diseñado para entrenamiento desde cero en tareas de clasificación multicategoría.
    """

    def __init__(self, num_classes=20):
        """
        Inicializa el modelo, definiendo su arquitectura y métricas.
        """
        super(ClassifierA, self).__init__()
        # Definición de la arquitectura del modelo
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Capa convolucional 1
            nn.ReLU(),
            nn.MaxPool2d(2),  # Reducción espacial
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Capa convolucional 2
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),  # Aplanar las dimensiones para la entrada a capas densas
            nn.Linear(128 * 32 * 32, 512),  # Capa completamente conectada
            nn.ReLU(),
            nn.Linear(512, num_classes)  # Capa de salida con `num_classes`
        )

        # Métricas para seguimiento durante entrenamiento, validación y prueba
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.recall = Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')

    def forward(self, x):
        """
        Propagación hacia adelante.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Paso de entrenamiento: calcula la pérdida y la registra.
        """
        x, y = batch
        y_hat = self(x)  # Predicciones del modelo
        loss = F.cross_entropy(y_hat, y)  # Cálculo de pérdida
        self.log('train_loss', loss)  # Registro de la pérdida
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Paso de validación: calcula pérdida y métricas.
        """
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)  # Pérdida en validación
        preds = torch.argmax(y_hat, dim=1)  # Predicciones más probables

        # Actualización de métricas
        acc = self.accuracy(preds, y)
        prec = self.precision(preds, y)
        rec = self.recall(preds, y)
        f1 = self.f1(preds, y)

        # Registro de métricas
        self.log('val_loss', val_loss)
        self.log('val_acc', acc)
        self.log('val_precision', prec)
        self.log('val_recall', rec)
        self.log('val_f1', f1)
        return val_loss

    def test_step(self, batch, batch_idx):
        """
        Paso de prueba: calcula la pérdida y métricas.
        """
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)  # Pérdida en prueba
        preds = torch.argmax(y_hat, dim=1)  # Predicciones más probables

        # Actualización y registro de métricas
        acc = self.accuracy(preds, y)
        prec = self.precision(preds, y)
        rec = self.recall(preds, y)
        f1 = self.f1(preds, y)

        self.log('test_loss', test_loss)
        self.log('test_acc', acc)
        self.log('test_precision', prec)
        self.log('test_recall', rec)
        self.log('test_f1', f1)
        return test_loss

    def configure_optimizers(self):
        """
        Configura el optimizador para el modelo.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-4)


# Clasificador B1: Modelo que utiliza un encoder preentrenado con pesos congelados
class ClassifierB1(pl.LightningModule):
    """
    Modelo clasificador que utiliza un encoder preentrenado con pesos congelados para extracción de características.
    """

    def __init__(self, encoder, num_classes=20):
        """
        Inicializa el modelo, definiendo su arquitectura y métricas.
        """
        super(ClassifierB1, self).__init__()
        self.encoder = encoder
        
        # Congela los pesos del encoder para que no se actualicen durante el entrenamiento
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Definición de la parte clasificadora del modelo
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Aplanar las características extraídas por el encoder
            nn.Linear(256 * 16 * 16, 512),  # Capa completamente conectada
            nn.ReLU(),
            nn.Linear(512, num_classes)  # Capa de salida
        )

        # Métricas para seguimiento durante entrenamiento, validación y prueba
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.recall = Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')

    def forward(self, x):
        """
        Propagación hacia adelante.
        """
        x = self.encoder(x)  # Extracción de características usando el encoder
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        """
        Paso de entrenamiento: calcula la pérdida y la registra.
        """
        x, y = batch
        y_hat = self(x)  # Predicciones del modelo
        loss = F.cross_entropy(y_hat, y)  # Cálculo de pérdida
        self.log('train_loss', loss)  # Registro de la pérdida
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Paso de validación: calcula pérdida y métricas.
        """
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)  # Pérdida en validación
        preds = torch.argmax(y_hat, dim=1)  # Predicciones más probables

        # Actualización de métricas
        acc = self.accuracy(preds, y)
        prec = self.precision(preds, y)
        rec = self.recall(preds, y)
        f1 = self.f1(preds, y)

        # Registro de métricas
        self.log('val_loss', val_loss)
        self.log('val_acc', acc)
        self.log('val_precision', prec)
        self.log('val_recall', rec)
        self.log('val_f1', f1)
        return val_loss

    def test_step(self, batch, batch_idx):
        """
        Paso de prueba: calcula la pérdida y métricas.
        """
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)  # Pérdida en prueba
        preds = torch.argmax(y_hat, dim=1)  # Predicciones más probables

        # Actualización y registro de métricas
        acc = self.accuracy(preds, y)
        prec = self.precision(preds, y)
        rec = self.recall(preds, y)
        f1 = self.f1(preds, y)

        self.log('test_loss', test_loss)
        self.log('test_acc', acc)
        self.log('test_precision', prec)
        self.log('test_recall', rec)
        self.log('test_f1', f1)
        return test_loss

    def configure_optimizers(self):
        """
        Configura el optimizador para el modelo.
        """
        return torch.optim.Adam(self.classifier.parameters(), lr=1e-4)  # Optimiza solo los parámetros de la parte clasificadora


# Clasificador B2: Modelo que utiliza un encoder preentrenado con pesos ajustables
class ClassifierB2(pl.LightningModule):
    """
    Modelo clasificador que utiliza un encoder preentrenado con pesos ajustables para extracción de características.
    """

    def __init__(self, encoder, num_classes=20):
        """
        Inicializa el modelo, definiendo su arquitectura y métricas.
        """
        super(ClassifierB2, self).__init__()
        self.encoder = encoder  # Encoder del autoencoder preentrenado
        # Arquitectura del clasificador
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Aplanar las características del encoder
            nn.Linear(256 * 16 * 16, 512),  # Capa completamente conectada
            nn.ReLU(),
            nn.Linear(512, num_classes)  # Capa de salida para predicción de clases
        )

        # Métricas para seguimiento del rendimiento
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.recall = Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')

    def forward(self, x):
        """
        Propagación hacia adelante.
        """
        x = self.encoder(x)  # Extrae características con el encoder
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        """
        Paso de entrenamiento: calcula la pérdida.
        """
        x, y = batch
        y_hat = self(x)  # Predicciones
        loss = F.cross_entropy(y_hat, y)  # Cálculo de la pérdida
        self.log('train_loss', loss)  # Registro de la pérdida
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Paso de validación: calcula la pérdida y las métricas.
        """
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)  # Pérdida en validación
        preds = torch.argmax(y_hat, dim=1)  # Predicciones más probables

        # Cálculo de métricas
        acc = self.accuracy(preds, y)
        prec = self.precision(preds, y)
        rec = self.recall(preds, y)
        f1 = self.f1(preds, y)

        # Registro de métricas
        self.log('val_loss', val_loss)
        self.log('val_acc', acc)
        self.log('val_precision', prec)
        self.log('val_recall', rec)
        self.log('val_f1', f1)
        return val_loss

    def test_step(self, batch, batch_idx):
        """
        Paso de prueba: calcula la pérdida y las métricas.
        """
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)  # Pérdida en prueba
        preds = torch.argmax(y_hat, dim=1)  # Predicciones más probables

        # Cálculo de métricas
        acc = self.accuracy(preds, y)
        prec = self.precision(preds, y)
        rec = self.recall(preds, y)
        f1 = self.f1(preds, y)

        # Registro de métricas
        self.log('test_loss', test_loss)
        self.log('test_acc', acc)
        self.log('test_precision', prec)
        self.log('test_recall', rec)
        self.log('test_f1', f1)
        return test_loss

    def configure_optimizers(self):
        """
        Configura el optimizador para el modelo.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-4)

