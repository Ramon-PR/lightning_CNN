import config
from dataset import RirDataModule
from model_system import CNNModule, CallbackLog_loss_per_epoch
from models import CNN_basic


import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import os


from lightning.pytorch.callbacks import Timer
timer = Timer(duration="00:12:00:00")

pl.seed_everything(42, workers=True)

if __name__ == "__main__":

    dm = RirDataModule(config.DATA_DIR, config.BATCH_SIZE, config.NUM_WORKERS)

    logger = TensorBoardLogger("tb_logs", name=config.SAVE_NAME, 
                               log_graph=True, default_hp_metric=False)

    early_stop_callback = EarlyStopping(
        monitor=config.VAR2MONITOR,
        patience=10,
        verbose=False,
        mode='min'
    )

    # TO load a checkpoint:
    # model = LitCNN.load_from_checkpoint(checkpoint_path="./checkpoints/model-???.ckpt")
    checkpoint = ModelCheckpoint(
        dirpath='./checkpoints/',         # where to save the checkpoint
        filename=config.SAVE_NAME,        # name of the model
        save_top_k=1,                     # save the best k models
        monitor=config.VAR2MONITOR,             # variable to monitor
        mode='min'                        # that has to be min or max
    )


    model_dict = {}
    model_dict["CNN_basic"] = CNN_basic
    CNN_model = CNNModule(
        model_name=config.MODEL_NAME,
        model_dict=model_dict,
        model_hparams={
            "n_channels" : config.N_CHANNELS ,
            "Hin" : config.HIN ,
            "Win" : config.WIN ,
            "Hout" : config.HOUT ,
            "Wout" : config.WOUT ,
            "n_filters" : config.N_FILTERS ,
            "conv_kern" : config.CONV_KERN ,
            "conv_pad" : config.CONV_PAD ,
            "act_fn_name" : "relu" ,
        },
        optimizer_name="Adam",
        optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
    )

    # If the restart file exist load it and run an extra NUM_EPOCHS
    # Otherwise start from scratch and run NUM_EPOCHS
    
    # temporal trainer object
    trainer_temp = pl.Trainer()
    
    restart_file = None
    max_epochs = config.NUM_EPOCHS

    path_restart_file = os.path.join(checkpoint.dirpath, checkpoint.filename+'.ckpt')
    if path_restart_file and os.path.isfile(path_restart_file):
        restart_file = path_restart_file
        # load checkpoint dictionary
        dic_ckpt = trainer_temp.strategy.load_checkpoint(restart_file)

        # Add extra epochs to train
        max_epochs = dic_ckpt['epoch'] + config.NUM_EPOCHS
        del trainer_temp, dic_ckpt


    # Set up the trainer with the number of epochs to train
    trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=max_epochs,
        precision=config.PRECISION,
        # callbacks=[early_stop_callback, timer, checkpoint, CallbackLog_loss_per_epoch()],
        callbacks=[timer, checkpoint, CallbackLog_loss_per_epoch()],
        logger=logger,
        deterministic=True,
        )

    # %% FIT / VALIDATE / TEST
    trainer.fit(CNN_model, dm, ckpt_path=restart_file)

    # trainer.validate(model, dm)
    # trainer.test(model, dm)


    # %%query training/validation/test time (in seconds)
    _stage = "train" #train, validate, test
    a = timer.start_time(_stage)
    b = timer.end_time(_stage)
    c = timer.time_elapsed(_stage)

    print(f"Time elapsed in last run  = {b-a}")
    print(f"Time elapsed in all runs  = {c}")


# resnetpreact_model, resnetpreact_results = train_model(
#     model_name="ResNet",
#     model_hparams={
#         "num_classes": 10,
#         "c_hidden": [16, 32, 64],
#         "num_blocks": [3, 3, 3],
#         "act_fn_name": "relu",
#         "block_name": "PreActResNetBlock",
#     },
#     optimizer_name="SGD",
#     optimizer_hparams={"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
#     save_name="ResNetPreAct",
# )


