import torch
import torch.nn as nn


act_fn_by_name = {"tanh": nn.Tanh, "relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "gelu": nn.GELU}

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



def blockMaxP(chan_in, chan_out, act_fn_name="relu", kernel_size=3, pad1=1, str1=1, kernel_pool=2, str_pool=2):
    return torch.nn.Sequential(
        torch.nn.Conv2d(chan_in, chan_out, kernel_size, padding=pad1, stride=str1),
        act_fn_by_name[act_fn_name](),
        torch.nn.MaxPool2d(kernel_pool, stride=str_pool)
    )


def blockMinP(chan_in, chan_out, act_fn_name="relu", kernel_size=3, pad1=1, str1=1, kernel_pool=2, str_pool=2):
    return torch.nn.Sequential(
        torch.nn.Conv2d(chan_in, chan_out, kernel_size, padding=pad1, stride=str1),
        act_fn_by_name[act_fn_name](),
        MinPool2d(kernel_pool, stride=str_pool)
    )


def blockLinear(c_in, c_out, act_fn_name="relu"):
    return torch.nn.Sequential(
        torch.nn.Linear(c_in, c_out),
        act_fn_by_name[act_fn_name](),
    )


def blockMaxP_BN(chan_in, chan_out, act_fn_name="relu", kernel_size=3, pad1=1, str1=1, kernel_pool=2, str_pool=2):
    return torch.nn.Sequential(
        torch.nn.Conv2d(chan_in, chan_out, kernel_size, padding=pad1, stride=str1),
        torch.nn.BatchNorm2d(chan_out),
        act_fn_by_name[act_fn_name](),
        torch.nn.MaxPool2d(kernel_pool, stride=str_pool)
    )


def blockMinP_BN(chan_in, chan_out, act_fn_name="relu", kernel_size=3, pad1=1, str1=1, kernel_pool=2, str_pool=2):
    return torch.nn.Sequential(
        torch.nn.Conv2d(chan_in, chan_out, kernel_size, padding=pad1, stride=str1),
        torch.nn.BatchNorm2d(chan_out),
        act_fn_by_name[act_fn_name](),
        MinPool2d(kernel_pool, stride=str_pool),
    )




class CNN_basic(torch.nn.Module):
    def __init__(self, n_channels: int = 1, 
        Hin: int = 32, Win: int = 32, 
        Hout: int = 32, Wout: int = 32,
        n_filters: list = [5, 5],
        conv_kern: list = [3, 5],
        conv_pad: list = [1, 2],
        act_fn_name="relu",
        ):

        super().__init__()

        chan_in = n_channels 

        n_filters1, n_filters2 = n_filters
        kernel_size1, kernel_size2 = conv_kern
        pad1, pad2 = conv_pad
        str1, str2 = 1, 1

        kernel_pool, str_pool = 2, 2

        self.Hin = Hin
        self.Win = Win
        self.Hout = Hout
        self.Wout = Wout

        n_outputs = self.Hout* self.Wout

        self.conv1 = blockMaxP(chan_in, n_filters1, act_fn_name, kernel_size1, pad1, str1)
        H = dim_after_filter(Hin, kernel_size1, pad1, str1)
        W = dim_after_filter(Win, kernel_size1, pad1, str1)
        # if maxpool2d
        H, W = dim_after_filter(H, kernel_pool, 0, str_pool), dim_after_filter(W, kernel_pool, 0, str_pool) 

        self.conv2 = blockMaxP(n_filters1, n_filters2, act_fn_name, kernel_size2, pad2, str2)
        H = dim_after_filter(H, kernel_size2, pad2, str2)
        W = dim_after_filter(W, kernel_size2, pad2, str2)
        # if maxpool2d
        H, W = dim_after_filter(H, kernel_pool, 0, str_pool), dim_after_filter(W, kernel_pool, 0, str_pool)

        self.fc = torch.nn.Linear(n_filters2*H*W, n_outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x




class CNN_2branch(torch.nn.Module):
    def __init__(self, n_channels: int = 1, 
        Hin: int = 32, Win: int = 32, 
        Hout: int = 32, Wout: int = 32,
        n_filters: list = [5, 5],
        conv_kern: list = [3, 5],
        conv_pad: list = [1, 2],
        act_fn_name="relu",
        ):

        super().__init__()

        chan_in = n_channels 

        n_filters1, n_filters2 = n_filters
        kernel_size1, kernel_size2 = conv_kern
        pad1, pad2 = conv_pad
        str1, str2 = 1, 1

        kernel_pool, str_pool = 2, 2

        self.Hin = Hin
        self.Win = Win
        self.Hout = Hout
        self.Wout = Wout

        n_outputs = self.Hout* self.Wout

        self.conv1 = blockMaxP(chan_in, n_filters1, act_fn_name, kernel_size1, pad1, str1)
        H1 = dim_after_filter(Hin, kernel_size1, pad1, str1)
        W1 = dim_after_filter(Win, kernel_size1, pad1, str1)
        # if maxpool2d
        H1, W1 = dim_after_filter(H1, kernel_pool, 0, str_pool), dim_after_filter(W1, kernel_pool, 0, str_pool) 

        self.conv2 = blockMinP(chan_in, n_filters2, act_fn_name, kernel_size2, pad2, str2)
        H2 = dim_after_filter(Hin, kernel_size2, pad2, str2)
        W2 = dim_after_filter(Win, kernel_size2, pad2, str2)
        # if minpool2d
        H2, W2 = dim_after_filter(H2, kernel_pool, 0, str_pool), dim_after_filter(W2, kernel_pool, 0, str_pool) 

        self.fc = torch.nn.Linear(n_filters1*H1*W1 + n_filters2*H2*W2, n_outputs)

    def forward(self, x): 
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        y = torch.cat((x1.view(x1.shape[0], -1), x2.view(x2.shape[0], -1)), -1)    

        y = self.fc(y)
        return y


model_dict={}
model_dict["CNN_basic"] = CNN_basic
model_dict["CNN_2branch"] = CNN_2branch








