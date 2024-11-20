import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score

# Clasificador A: Entrenado desde cero
class ClassifierA(pl.LightningModule):
    def __init__(self, num_classes=20):
        super(ClassifierA, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        # Define las métricas con el argumento 'task'
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.recall = Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)

        # Actualiza las métricas
        acc = self.accuracy(preds, y)
        prec = self.precision(preds, y)
        rec = self.recall(preds, y)
        f1 = self.f1(preds, y)

        # Registra las métricas
        self.log('val_loss', val_loss)
        self.log('val_acc', acc)
        self.log('val_precision', prec)
        self.log('val_recall', rec)
        self.log('val_f1', f1)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        
        # Calculate metrics
        acc = self.accuracy(preds, y)
        prec = self.precision(preds, y)
        rec = self.recall(preds, y)
        f1 = self.f1(preds, y)

        # Log metrics
        self.log('test_loss', test_loss)
        self.log('test_acc', acc)
        self.log('test_precision', prec)
        self.log('test_recall', rec)
        self.log('test_f1', f1)
        return test_loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

# Clasificador B1: Usando el encoder del Autoencoder con pesos congelados
class ClassifierB1(pl.LightningModule):
    def __init__(self, encoder, num_classes=20):
        super(ClassifierB1, self).__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False  # Congela los pesos del encoder
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        # Define las métricas
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.recall = Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)

        # Actualiza las métricas
        acc = self.accuracy(preds, y)
        prec = self.precision(preds, y)
        rec = self.recall(preds, y)
        f1 = self.f1(preds, y)

        # Registra las métricas
        self.log('val_loss', val_loss)
        self.log('val_acc', acc)
        self.log('val_precision', prec)
        self.log('val_recall', rec)
        self.log('val_f1', f1)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        
        # Calculate metrics
        acc = self.accuracy(preds, y)
        prec = self.precision(preds, y)
        rec = self.recall(preds, y)
        f1 = self.f1(preds, y)

        # Log metrics
        self.log('test_loss', test_loss)
        self.log('test_acc', acc)
        self.log('test_precision', prec)
        self.log('test_recall', rec)
        self.log('test_f1', f1)
        return test_loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=1e-4)

# Clasificador B2: Usando el encoder del Autoencoder con pesos ajustables
class ClassifierB2(pl.LightningModule):
    def __init__(self, encoder, num_classes=20):
        super(ClassifierB2, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        # Define las métricas
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.recall = Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)

        # Actualiza las métricas
        acc = self.accuracy(preds, y)
        prec = self.precision(preds, y)
        rec = self.recall(preds, y)
        f1 = self.f1(preds, y)

        # Registra las métricas
        self.log('val_loss', val_loss)
        self.log('val_acc', acc)
        self.log('val_precision', prec)
        self.log('val_recall', rec)
        self.log('val_f1', f1)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        
        # Calculate metrics
        acc = self.accuracy(preds, y)
        prec = self.precision(preds, y)
        rec = self.recall(preds, y)
        f1 = self.f1(preds, y)

        # Log metrics
        self.log('test_loss', test_loss)
        self.log('test_acc', acc)
        self.log('test_precision', prec)
        self.log('test_recall', rec)
        self.log('test_f1', f1)
        return test_loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
