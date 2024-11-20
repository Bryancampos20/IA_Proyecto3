import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import wandb
from pytorch_lightning.loggers import WandbLogger
from unet_autoencoder import UNetAutoencoder
import importlib
import butterfly_data_module

# Recarga el módulo de datos en caso de modificaciones
importlib.reload(butterfly_data_module)
from butterfly_data_module import ButterflyDataModule

# Configuración de Weights and Biases (WandB) para seguimiento del entrenamiento
wandb_logger = WandbLogger(project='butterfly_autoencoder')

# Instancia el DataModule que gestiona los datos de entrenamiento y prueba
data_module = ButterflyDataModule(data_dir='/home/miguel/Desktop/IA_Proyeto_3/archive/')
data_module.setup()

# Crea una instancia del modelo Autoencoder basado en U-Net
autoencoder = UNetAutoencoder()

# Configura EarlyStopping para detener el entrenamiento si la pérdida no mejora
early_stopping = EarlyStopping(
    monitor='train_loss',  # Supervisa la pérdida de entrenamiento
    patience=7,            # Número de épocas sin mejora antes de detener
    mode='min'             # Busca minimizar la pérdida
)

# Configura el Trainer de PyTorch Lightning
trainer = Trainer(
    max_epochs=30,          # Número máximo de épocas para entrenar
    logger=wandb_logger,    # Logger para registrar métricas en WandB
    callbacks=[early_stopping],  # Callback para detener entrenamiento temprano
    accelerator='cuda',     # Usa GPU para acelerar el entrenamiento
    devices=1,              # Número de GPUs a usar
    log_every_n_steps=5     # Frecuencia de registro de métricas
)

# Inicia el entrenamiento del Autoencoder
trainer.fit(autoencoder, data_module)

# Guarda el modelo entrenado en un archivo
trainer.save_checkpoint("/home/miguel/Desktop/IA_Proyeto_3/autoencoder.ckpt")
