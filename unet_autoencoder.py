import torch
import torch.nn as nn
import pytorch_lightning as pl

class UNetAutoencoder(pl.LightningModule):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()
        
        # Encoder: Reduce la dimensionalidad mientras extrae características
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Convolución inicial
            nn.ReLU(),  # Activación
            nn.MaxPool2d(2),  # Reduce el tamaño a la mitad
            nn.Dropout(0.3)  # Previene sobreajuste
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        
        # Agrupamos las capas del encoder en un módulo secuencial
        self.encoder = nn.Sequential(
            self.encoder1,
            self.encoder2,
            self.encoder3
        )

        # Decoder: Reconstruye la imagen original a partir de las características
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # Aumenta la resolución
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid()  # Escala los valores a [0, 1] para representar imágenes
        )

    def forward(self, x):
        # Paso de codificación
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        
        # Paso de decodificación con conexiones de salto
        dec3 = self.decoder3(enc3) + enc2  # Suma las características del encoder
        dec2 = self.decoder2(dec3) + enc1
        dec1 = self.decoder1(dec2)
        
        return dec1  # Salida reconstruida
    
    def training_step(self, batch, batch_idx):
        x, _ = batch  # Solo necesitamos las imágenes
        x_hat = self.forward(x)  # Reconstrucción de la imagen
        loss = nn.MSELoss()(x_hat, x)  # Calcula el error cuadrático medio
        self.log('train_loss', loss)  # Registra la pérdida de entrenamiento
        return loss
    
    def configure_optimizers(self):
        # Adam optimizer con una tasa de aprendizaje inicial
        return torch.optim.Adam(self.parameters(), lr=5e-4)
