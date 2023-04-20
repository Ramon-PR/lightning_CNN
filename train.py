import config
from dataset import RirDataModule
from model import LitCNN
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger


from lightning.pytorch.callbacks import Timer
timer = Timer(duration="00:12:00:00")

if __name__ == "__main__":

    model = LitCNN(config.N_CHANNELS, config.HIN, config.WIN,
                   config.HOUT, config.WOUT, config.LEARNING_RATE)

    dm = RirDataModule(config.DATA_DIR, config.BATCH_SIZE, config.NUM_WORKERS)

    logger = TensorBoardLogger("tb_logs", name="my_model")

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=False,
        mode='min'
    )
    
    # %% TRAINING Set Up
    # Set the training options:
    # -------------------------
    trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
        callbacks=[early_stop_callback, timer],
        logger=logger,
        )

    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)

# %%

# query training/validation/test time (in seconds)
_stage = "test" #train, validate, test
a = timer.start_time(_stage)
b = timer.end_time(_stage)
c = timer.time_elapsed(_stage)

print(f"end_time-start_time  = {b-a}")
print(f"time_elapsed         = {c}")

