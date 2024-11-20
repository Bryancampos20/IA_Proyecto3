import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from unet_autoencoder import UNetAutoencoder
import importlib
import butterfly_data_module
importlib.reload(butterfly_data_module)
from butterfly_data_module import ButterflyDataModule
from classifier import ClassifierA, ClassifierB1, ClassifierB2

# Define el directorio donde se guardarán los checkpoints
ckpt_dir = '/home/miguel/Desktop/IA_Proyeto_3/checkpoints/'
os.makedirs(ckpt_dir, exist_ok=True)

# Configura Weights and Biases con loggers separados
wandb_logger_a = WandbLogger(project='butterfly_classification', name='ClassifierA')
wandb_logger_b1 = WandbLogger(project='butterfly_classification', name='ClassifierB1')
wandb_logger_b2 = WandbLogger(project='butterfly_classification', name='ClassifierB2')

# Instancia el DataModule
data_module = ButterflyDataModule(data_dir='/home/miguel/Desktop/IA_Proyeto_3/archive/', batch_size=64)
data_module.setup()

# Carga el Autoencoder entrenado
autoencoder = UNetAutoencoder.load_from_checkpoint('/home/miguel/Desktop/IA_Proyeto_3/autoencoder.ckpt')
encoder = autoencoder.encoder

# Instancia los clasificadores
classifier_a = ClassifierA(num_classes=20)
classifier_b1 = ClassifierB1(encoder, num_classes=20)
classifier_b2 = ClassifierB2(encoder, num_classes=20)

# Configura EarlyStopping
early_stopping = EarlyStopping(monitor='train_loss', patience=5, mode='min')

# Configura ModelCheckpoint para guardar los modelos
checkpoint_callback_a = ModelCheckpoint(
    dirpath=ckpt_dir,
    monitor='train_loss',
    filename='classifier_a-{epoch:02d}-{train_loss:.2f}',
    save_top_k=1,
    mode='min'
)

checkpoint_callback_b1 = ModelCheckpoint(
    dirpath=ckpt_dir,
    monitor='train_loss',
    filename='classifier_b1-{epoch:02d}-{train_loss:.2f}',
    save_top_k=1,
    mode='min'
)

checkpoint_callback_b2 = ModelCheckpoint(
    dirpath=ckpt_dir,
    monitor='train_loss',
    filename='classifier_b2-{epoch:02d}-{train_loss:.2f}',
    save_top_k=1,
    mode='min'
)

# Entrena ClassifierA
trainer_a = Trainer(
    max_epochs=30,
    logger=wandb_logger_a,
    callbacks=[early_stopping, checkpoint_callback_a],
    accelerator='cuda',
    devices=1,
    log_every_n_steps=5
)
trainer_a.fit(classifier_a, data_module)
wandb.finish()

# Entrena ClassifierB1
trainer_b1 = Trainer(
    max_epochs=30,
    logger=wandb_logger_b1,
    callbacks=[early_stopping, checkpoint_callback_b1],
    accelerator='cuda',
    devices=1,
    log_every_n_steps=5
)
trainer_b1.fit(classifier_b1, data_module)
wandb.finish()

# Entrena ClassifierB2
trainer_b2 = Trainer(
    max_epochs=30,
    logger=wandb_logger_b2,
    callbacks=[early_stopping, checkpoint_callback_b2],
    accelerator='cuda',
    devices=1,
    log_every_n_steps=5
)
trainer_b2.fit(classifier_b2, data_module)
wandb.finish()
