
import config
from dataset import RirDataModule
from model import LitCNN

import lightning.pytorch as pl

from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger



if __name__ == "__main__":

	model = LitCNN(config.N_CHANNELS, config.HIN, config.WIN, config.HOUT, config.WOUT, config.LEARNING_RATE)

	dm = RirDataModule(config.DATA_DIR, config.BATCH_SIZE, config.NUM_WORKERS)

	logger = TensorBoardLogger("tb_logs", name="my_model")

	# %% TRAINING Set Up
	# Set the training options:
	# -------------------------
	trainer = pl.Trainer(
		accelerator = config.ACCELERATOR,
		devices = config.DEVICES,
		min_epochs = 1,
		max_epochs = config.NUM_EPOCHS,
		precision = config.PRECISION,
		callbacks = [EarlyStopping(monitor="val_loss")],
		logger=logger,
		)

	trainer.fit(model, dm)
	trainer.validate(model, dm)
	trainer.test(model, dm)



