import torch
import torch.nn as nn
import pytorch_lightning as pl

class UNetAutoencoder(pl.LightningModule):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()
        
        # Define las capas del encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
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
        
        # Combina las capas del encoder en un solo módulo
        self.encoder = nn.Sequential(
            self.encoder1,
            self.encoder2,
            self.encoder3
        )

        # Define las capas del decoder
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        
        dec3 = self.decoder3(enc3) + enc2
        dec2 = self.decoder2(dec3) + enc1
        dec1 = self.decoder1(dec2)
        
        return dec1
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        loss = nn.MSELoss()(x_hat, x)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-4)
