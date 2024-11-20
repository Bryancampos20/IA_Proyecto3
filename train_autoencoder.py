import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import wandb
from pytorch_lightning.loggers import WandbLogger
from unet_autoencoder import UNetAutoencoder
import importlib
import butterfly_data_module
importlib.reload(butterfly_data_module)
from butterfly_data_module import ButterflyDataModule



# Configura Weights and Biases
wandb_logger = WandbLogger(project='butterfly_autoencoder')

# Instancia el DataModule
data_module = ButterflyDataModule(data_dir='/home/miguel/Desktop/IA_Proyeto_3/archive/')
data_module.setup()

# Instancia el Autoencoder
autoencoder = UNetAutoencoder()

# Configura EarlyStopping para evitar el sobreajuste
early_stopping = EarlyStopping(monitor='train_loss', patience=7, mode='min')

# Crea el Trainer
trainer = Trainer(
    max_epochs=30,
    logger=wandb_logger,
    callbacks=[early_stopping],
    accelerator='cuda',
    devices=1,
    log_every_n_steps=5
)

# Entrena el Autoencoder
trainer.fit(autoencoder, data_module)

# Guarda el modelo entrenado
trainer.save_checkpoint("/home/miguel/Desktop/IA_Proyeto_3/autoencoder.ckpt")
