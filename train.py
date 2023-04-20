import config
from dataset import RirDataModule
from model import LitCNN
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
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

    # TO load a checkpoint:
    # model = LitCNN.load_from_checkpoint(checkpoint_path="./checkpoints/model-???.ckpt")
    checkpoint = ModelCheckpoint(
        dirpath='./checkpoints/',         # where to save the checkpoint
        filename='model-{val_loss:.5f}',  # name of the model
        save_top_k=1,                     # save the best k models
        monitor='val_loss',               # variable to monitor
        mode='min'                        # that has to be min or max
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
        callbacks=[early_stop_callback, timer, checkpoint],
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

