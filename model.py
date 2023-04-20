import lightning.pytorch as pl
import torch
import torchmetrics

def dim_after_filter(dim_in, dim_kernel, pad, stripe):
    return int((dim_in + 2*pad - dim_kernel)/stripe) + 1


class MinPool2d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False):
        super().__init__()
  
        self.kernel_size, self.stride = kernel_size, stride 
        self.padding, self.dilation = padding, dilation
        self.ceil_mode = ceil_mode
  
    def forward(self, x):
        x = -torch.nn.MaxPool2d(self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)(-x)
        return x
    
    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'kernel_size={}, stride={}, padding={}, dilation={}, ceil_mode={})'.format(
            self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode is not None)


def blockRelU(chan_in, chan_out, kernel_size=3, pad1=1, str1=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(chan_in, chan_out, kernel_size, padding=pad1, stride=str1),
        torch.nn.ReLU()
    )

def blockRelUMaxP(chan_in, chan_out, kernel_size=3, pad1=1, str1=1, kernel_maxpool=2, str_maxpool=2):
    return torch.nn.Sequential(
        torch.nn.Conv2d(chan_in, chan_out, kernel_size, padding=pad1, stride=str1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_maxpool, stride=str_maxpool)
    )

def blockRelUMinP(chan_in, chan_out, kernel_size=3, pad1=1, str1=1, kernel_minpool=2, str_minpool=2):
    return torch.nn.Sequential(
        torch.nn.Conv2d(chan_in, chan_out, kernel_size, padding=pad1, stride=str1),
        torch.nn.ReLU(),
        MinPool2d(kernel_minpool, stride=str_minpool)
    )

    

class LitCNN(pl.LightningModule):
    def __init__(self, chan_in=1, Hin=32, Win=32, Hout=32, Wout=32, learning_rate=1e-3):
        super().__init__()

        n_filters1, n_filters2 = 5, 5
        kernel_size1, kernel_size2 = 3, 5
        pad1, pad2 = 1, 2
        str1, str2 = 1, 1

        kernel_maxpool, str_maxpool = 2, 2
        self.criterion = torchmetrics.MeanSquaredError()
        self.lr = learning_rate
        self.Hin = Hin
        self.Win = Win
        self.Hout = Hout
        self.Wout = Wout
        
        n_outputs = self.Hout* self.Wout

        self.conv1 = blockRelUMaxP(chan_in, n_filters1, kernel_size1, pad1, str1)
        H = dim_after_filter(Hin, kernel_size1, pad1, str1)
        W = dim_after_filter(Win, kernel_size1, pad1, str1)
        # if maxpool2d
        H, W = dim_after_filter(H, kernel_maxpool, 0, str_maxpool), dim_after_filter(W, kernel_maxpool, 0, str_maxpool) 

        self.conv2 = blockRelUMaxP(n_filters1, n_filters2, kernel_size2, pad2, str2)
        H = dim_after_filter(H, kernel_size2, pad2, str2)
        W = dim_after_filter(W, kernel_size2, pad2, str2)
        # if maxpool2d
        H, W = dim_after_filter(H, kernel_maxpool, 0, str_maxpool), dim_after_filter(W, kernel_maxpool, 0, str_maxpool)
        
        self.fc = torch.nn.Linear(n_filters2*H*W, n_outputs)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _common_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        image, target, submask = batch
        target = target.view(target.shape[0],-1)
        y_pred = self(image)
        
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
