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


def blockC(chan_in, chan_out, act_fn_name="relu", kernel_size=3, pad1=1, str1=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(chan_in, chan_out, kernel_size, padding=pad1, stride=str1),
        act_fn_by_name[act_fn_name](),
    )


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
        conv_str: list = [1, 1],
        act_fn_name="relu",
        ):

        super().__init__()

        kernel_pool, str_pool = 2, 2

        self.Hin = Hin
        self.Win = Win
        self.Hout = Hout
        self.Wout = Wout
        self.n_filters = n_filters
        self.conv_kernels = conv_kern
        self.conv_pad = conv_pad
        self.conv_str = conv_str

        n_outputs = self.Hout* self.Wout

        num_layers = len(self.conv_kernels)
        
        layers = []
        self.layer_cin = []
        self.layer_cin.append(n_channels)
        self.layer_cin += self.n_filters[:-1]

        H, W = self.Hin, self.Win
        for layer_idx in range(num_layers):
            layers.append( blockMaxP( self.layer_cin[layer_idx], self.n_filters[layer_idx], act_fn_name,
                                     self.conv_kernels[layer_idx], self.conv_pad[layer_idx], self.conv_str[layer_idx])
                          )
            H = dim_after_filter(H, self.conv_kernels[layer_idx], self.conv_pad[layer_idx], self.conv_str[layer_idx])
            W = dim_after_filter(W, self.conv_kernels[layer_idx], self.conv_pad[layer_idx], self.conv_str[layer_idx])
            # if maxpool2d
            H = dim_after_filter(H, kernel_pool, 0, str_pool)
            W = dim_after_filter(W, kernel_pool, 0, str_pool)
            
        self.block_layers = nn.Sequential(*layers)

        self.fc = torch.nn.Linear(self.n_filters[-1]*H*W, n_outputs)

    def forward(self, x):
        x = self.block_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = x.view(-1, 1, self.Hout, self.Wout)
        return x



class CNN_basic_cT(torch.nn.Module):
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
        self.chan_out = chan_in
        self.n_filters1 = n_filters1
        self.n_filters2 = n_filters2
        

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

        self.convTr1 = torch.nn.ConvTranspose2d(
            in_channels = n_filters2, 
            out_channels = n_filters1, 
            kernel_size = kernel_pool,
            stride=kernel_pool)
        
        self.convTr2 = torch.nn.ConvTranspose2d(
            in_channels = n_filters1, 
            out_channels = chan_in, 
            kernel_size = kernel_pool,
            stride=kernel_pool)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.convTr1(x, output_size = [-1, self.n_filters1, int(self.Hout/2), int(self.Wout/2)])
        x = self.convTr2(x, output_size = [-1, self.chan_out, self.Hout, self.Wout])
        return x



class CNN_basic_cT2(torch.nn.Module):
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
        self.chan_out = chan_in
        self.n_filters1 = n_filters1
        self.n_filters2 = n_filters2
        

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

        self.convTr = torch.nn.ConvTranspose2d(
            in_channels = n_filters2, 
            out_channels = self.chan_out, 
            kernel_size = kernel_pool*2,
            stride=str_pool*2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.convTr(x, output_size = [-1, self.chan_out, self.Hout, self.Wout])
        return x

class CNN_2branch(torch.nn.Module):
    def __init__(self, n_channels: int = 1, 
        Hin: int = 32, Win: int = 32, 
        Hout: int = 32, Wout: int = 32,
        n_filters: list = [5, 5],
        conv_kern: list = [3, 5],
        conv_pad: list = [1, 2],
        conv_str: list = [1, 1],
        act_fn_name="relu",
        ):

        super().__init__()

        kernel_pool, str_pool = 2, 2

        self.Hin = Hin
        self.Win = Win
        self.Hout = Hout
        self.Wout = Wout
        self.n_filters = n_filters
        self.conv_kernels = conv_kern
        self.conv_pad = conv_pad
        self.conv_str = conv_str

        n_outputs = self.Hout* self.Wout

        num_layers = len(self.conv_kernels)
        
        layersB1 = []
        layersB2 = []
        self.layer_cin = []
        self.layer_cin.append(n_channels)
        self.layer_cin += self.n_filters[:-1]

        H, W = self.Hin, self.Win
        for layer_idx in range(num_layers):
            layersB1.append( blockMaxP( self.layer_cin[layer_idx], self.n_filters[layer_idx], act_fn_name,
                                     self.conv_kernels[layer_idx], self.conv_pad[layer_idx], self.conv_str[layer_idx])
                          )
            layersB2.append( blockMinP( self.layer_cin[layer_idx], self.n_filters[layer_idx], act_fn_name,
                                     self.conv_kernels[layer_idx], self.conv_pad[layer_idx], self.conv_str[layer_idx])
                          )
            H = dim_after_filter(H, self.conv_kernels[layer_idx], self.conv_pad[layer_idx], self.conv_str[layer_idx])
            W = dim_after_filter(W, self.conv_kernels[layer_idx], self.conv_pad[layer_idx], self.conv_str[layer_idx])
            # if maxpool2d
            H = dim_after_filter(H, kernel_pool, 0, str_pool)
            W = dim_after_filter(W, kernel_pool, 0, str_pool)
            
        self.branch_1 = nn.Sequential(*layersB1)
        self.branch_2 = nn.Sequential(*layersB2)

        self.fc = torch.nn.Linear(self.n_filters[-1]*H*W + self.n_filters[-1]*H*W, n_outputs)

    def forward(self, x): 
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)

        y = torch.cat((x1.view(x1.shape[0], -1), x2.view(x2.shape[0], -1)), -1)    

        y = self.fc(y)
        y = y.view(-1, 1, self.Hout, self.Wout)
        return y



class CNN_2B(torch.nn.Module):
    def __init__(self, n_channels: int = 1, 
        Hin: int = 32, Win: int = 32, 
        Hout: int = 32, Wout: int = 32,
        n_filtersB1: list = [5, 5],
        n_filtersB2: list = [5, 5],
        conv_kernB1: list = [3, 3],
        conv_kernB2: list = [5, 5],
        conv_padB1: list = [1, 1],
        conv_padB2: list = [2, 2],
        conv_str: list = [1, 1],
        act_fn_name="relu",
        ):

        super().__init__()

        self.Hin = Hin
        self.Win = Win
        self.Hout = Hout
        self.Wout = Wout
        self.n_filtersB1 = n_filtersB1
        self.n_filtersB2 = n_filtersB2
        self.conv_kernelsB1 = conv_kernB1
        self.conv_kernelsB2 = conv_kernB2
        self.conv_padB1 = conv_padB1
        self.conv_padB2 = conv_padB2
        self.conv_str = conv_str

        n_outputs = self.Hout* self.Wout

        num_layers = len(self.conv_kernelsB1)
        
        layersB1 = []
        layersB2 = []
        self.layer_cinB1 = []
        self.layer_cinB1.append(n_channels)
        self.layer_cinB1 += self.n_filtersB1[:-1]
        self.layer_cinB2 = []
        self.layer_cinB2.append(n_channels)
        self.layer_cinB2 += self.n_filtersB2[:-1]

        H1, W1 = self.Hin, self.Win
        H2, W2 = self.Hin, self.Win
        for layer_idx in range(num_layers):
            layersB1.append( blockC( self.layer_cinB1[layer_idx], self.n_filtersB1[layer_idx], act_fn_name,
                                     self.conv_kernelsB1[layer_idx], self.conv_padB1[layer_idx], self.conv_str[layer_idx])
                          )
            layersB2.append( blockC( self.layer_cinB2[layer_idx], self.n_filtersB2[layer_idx], act_fn_name,
                                     self.conv_kernelsB2[layer_idx], self.conv_padB2[layer_idx], self.conv_str[layer_idx])
                          )
            H1 = dim_after_filter(H1, self.conv_kernelsB1[layer_idx], self.conv_padB1[layer_idx], self.conv_str[layer_idx])
            W1 = dim_after_filter(W1, self.conv_kernelsB1[layer_idx], self.conv_padB1[layer_idx], self.conv_str[layer_idx])
            H2 = dim_after_filter(H2, self.conv_kernelsB2[layer_idx], self.conv_padB2[layer_idx], self.conv_str[layer_idx])
            W2 = dim_after_filter(W2, self.conv_kernelsB2[layer_idx], self.conv_padB2[layer_idx], self.conv_str[layer_idx])
            
        self.H1, self.W1 = H1, W1
        self.H2, self.W2 = H2, W2
    
        self.branch_1 = nn.Sequential(*layersB1)
        self.branch_2 = nn.Sequential(*layersB2)

        self.fc = torch.nn.Linear(self.n_filtersB1[-1]*H1*W1 + self.n_filtersB2[-1]*H2*W2, n_outputs)

    def forward(self, x): 
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)

        y = torch.cat((x1.view(x1.shape[0], -1), x2.view(x2.shape[0], -1)), -1)    

        y = self.fc(y)
        y = y.view(-1, 1, self.Hout, self.Wout)
        return y

model_dict={}
model_dict["CNN_basic"] = CNN_basic
model_dict["CNN_2branch"] = CNN_2branch
model_dict["CNN_basic_cT"] = CNN_basic_cT
model_dict["CNN_basic_cT2"] = CNN_basic_cT2
model_dict["CNN_2B"] = CNN_2B







