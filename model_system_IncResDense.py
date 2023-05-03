import lightning.pytorch as pl
import torch
import torchmetrics
import config


# Function for setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def create_model(model_name, model_dict, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'

class VisionModule(pl.LightningModule):
    def __init__(self, model_name, model_dict, model_hparams, optimizer_name, optimizer_hparams):
        """
        Args:
            model_name: Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams: Hyperparameters for the model, as dictionary.
            optimizer_name: Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams: Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """

        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = create_model(model_name, model_dict, model_hparams)
        # Create loss module
        self.criterion = torchmetrics.MeanSquaredError()        
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, config.N_CHANNELS, config.HIN, config.WIN), dtype=torch.float32)
        self.training_step_outputs = []

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = torch.optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer]#, [scheduler]

    def _common_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        image, target, submask = batch
        y_pred = self.model(image)
        
        loss = self.criterion(y_pred, target)
        return loss


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        loss = self._common_step(batch, batch_idx)
        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=False ,on_epoch=True)
        # self.log_dict({'train_loss': loss}, 
                      # on_step=True, on_epoch=True, 
                      # prog_bar=True, logger=True)
        self.training_step_outputs.append(loss)
        
        return loss


    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        # it is independent of forward
        loss = self._common_step(batch, batch_idx)
        
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)
        return loss


    def test_step(self, batch, batch_idx):
        # test_step defines the test loop.
        # it is independent of forward
        loss = self._common_step(batch, batch_idx)
        
        # Logging to TensorBoard (if installed) by default
        self.log("test_loss", loss)
        return loss


class CallbackLog_loss_per_epoch(pl.Callback):
    
    def custom_histogram_adder(self, trainer, pl_module):
        # iterating through all parameters
        for name,params in pl_module.named_parameters():
            trainer.logger.experiment.add_histogram(name,params,trainer.current_epoch)
    
    def on_train_epoch_end(self, trainer, pl_module):
        # do something with all training_step outputs, for example:
        epoch_mean = torch.stack(pl_module.training_step_outputs).mean()
        # pl_module.log("training_epoch_mean", epoch_mean)
        trainer.logger.experiment.add_scalar("Loss/Train", epoch_mean, trainer.current_epoch)
        
        # logging histograms
        self.custom_histogram_adder(trainer, pl_module)
        
        # free up the memory
        pl_module.training_step_outputs.clear()
        # if(trainer.current_epoch==1):
            # sampleImg=torch.rand((1,1,config.HIN,config.WIN))
            # trainer.logger.experiment.add_graph(pl_module, sampleImg)

