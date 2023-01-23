import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import enum

from typing import Callable, List, Tuple, Optional
from torch.nn.modules import module

class PFNet(nn.Module):
    """
    Pre-defined Filter Network (PFNet).
    PFNet is a residual network with depthwise convolution, where the n x n convolution operations with n > 1 are pre-defined.
    
    n_classes determines the number of classes of the classification problem.
    
    start_config is a dictionary with information about the first Pre-defined Filter Module (PFM).
    k is the size of the pre-defined filters.
    filter_mode is one of Even, Uneven, All, Random and Spooth. It determines the type of pre-defined filters.
    n_angles is the number of different filter orientations. A value of 4 and filter_mode Uneven means that 4 different uneven edge filters are generated, i.e. left, top, right and bottom.
    n_channels_in and n_channels_out are the number of input and output channels.
    stride determines the stride of the pre-defined convolution operation.
    f determines the width of the module. f * n_channels_in intermediate channels are created internally.

    The subsequent blocks of the network are determined by a list with dictionaries, where each dict defines one DoublePredefinedFilterModule.
    Each block contains two PFMs and the filter modes can be assigned individually.
    n_blocks is the number of DoublePredefinedFilterModule to stack.
    predefined_filters_require_grad sets wether the pre-defined kernels should be adjusted during training.
    """
    def __init__(self,
        n_classes: int = 102,
        start_config: dict = {
            'k' : 3, 
            'filter_mode' : 'All',
            'n_angles' : 4,
            'n_channels_in' : 3,
            'n_channels_out' : 64,
            'stride' : 2,
            'f' : 16,
            'predefined_filters_require_grad' : False,
        },
        blockconfig_list: List[dict] = [
            {'k' : 3, 
            'filter_mode_1' : 'All',
            'filter_mode_2' : 'All',
            'n_angles' : 4,
            'n_blocks' : 2,
            'n_channels_in' : max(64, 64 * 2**(i-1)),
            'n_channels_out' : 64 * 2**i,
            'stride' : 2 if i>0 else 1,
            'f' : 1,
            'predefined_filters_require_grad' : False,
            } for i in range(4)],
        activation: str = 'relu',
        init_mode: str = 'kaiming_normal',
        statedict: str = '',
        ):
        super().__init__()

        # Create and store the actual model.
        self.embedded_model = PFNet_(
            n_classes=n_classes,
            start_config=start_config,
            blockconfig_list=blockconfig_list,
            init_mode=init_mode,
            activation=activation)
        
        # Load model weights.
        if statedict:
            pretrained_dict = torch.load(statedict, map_location=torch.device('cpu'))
            missing = self.load_state_dict(pretrained_dict, strict=True)
            print('Loading weights from statedict. Missing and unexpected keys:')
            print(missing)
                

    def forward(self, batch):
        if isinstance(batch, dict) and 'data' in batch:
            logits = self.embedded_model(batch['data'])
            out = {'logits' : logits}
            return out
        else:
            return self.embedded_model(batch)

            
    def save(self, statedict_name):
        torch.save(self.state_dict(), statedict_name)


class ParameterizedFilterMode(enum.Enum):
    # Enum for different pre-defined filters. 
    Even = 0 # Creates even edge filters.
    Uneven = 1 # Creaties uneven edge filters.
    All = 2 # Creates both even and uneven edge filters.
    Random = 3 # Creates filters drawn from the uniform distribution.
    Smooth = 4 # Creates a blur filter.


class PredefinedConv(nn.Module):
    def __init__(self, n_channels_in: int, n_channels_out: int, stride: int = 1) -> None:
        super().__init__()

        self.padding: int = 1
        self.weight: nn.Parameter = None
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        assert self.n_channels_out >= self.n_channels_in
        self.stride = stride
        self.internal_weight = None # set by subclass
        self.filter_requires_grad = None # set by subclass

    
    def forward(self, x: Tensor) -> Tensor:
        groups = self.n_channels_in
        if self.filter_requires_grad and self.training:
            # The internal weights have to be recomputed, because the weights cound have been changed.
            n_channels_per_kernel = self.n_channels_out // self.n_kernels
            self.internal_weight.data = self.weight.data.repeat((n_channels_per_kernel, 1, 1, 1)) 
        out = F.conv2d(x, self.internal_weight, None, self.stride, groups=groups, padding=self.padding)
        #print(f"multadds {x.shape[2]*x.shape[3]*self.n_channels_out*self.w.shape[1]*self.w.shape[2]}")
        return out


def saddle(x, y, phi, sigma, uneven=True):
    # Create an edge filter of certain angle phi and coordinates x and y.
    # The outline of a Gaussian distribution is used.
    # This outline is then oriented in 2D space to point to angle phi.
    a = np.arctan2(y, x)
    phi = np.deg2rad(phi)
    a = np.abs(phi - a)

    r = np.sqrt(x**2 + y**2)
    c = np.cos(a) * r

    if uneven:
        out = 1 - np.exp(-0.5*(c/sigma)**2)
        out[a>0.5*np.pi] = -out[a>0.5*np.pi]
    else:
        out = 2. * np.exp(-0.5*(c/sigma)**2)

    return out


def smooth(x, y, sigma):
    # Create a Gaussian filter kernel.
    r = np.sqrt(x**2 + y**2)
    out =  np.exp(-0.5*(r/sigma)**2)
    return out


def get_parameterized_filter(k: int=3, filter_mode: ParameterizedFilterMode=None, phi:float=0.):
    # Create a filter kernel of desired ParameterizedFilterMode type.
    # k determines the size of the kernel.
    # phi determines the angle (orientation) of the edge filter. 
    border = 0.5*(k-1.)
    x = np.linspace(-border, border, k)
    y = np.linspace(-border, border, k)
    xx, yy = np.meshgrid(x, y)
    if filter_mode==ParameterizedFilterMode.Even:
        data = saddle(xx, yy, phi, sigma=0.15*k, uneven=False)
        # Normalize the filter to have a mean of 0.
        data = data - data.mean()
    elif filter_mode==ParameterizedFilterMode.Uneven:
        data = saddle(xx, yy, phi, sigma=0.3*k, uneven=True)
        # Normalize the filter to have a mean of 0.
        data = data - data.mean()
    elif filter_mode==ParameterizedFilterMode.Smooth:
        data = smooth(xx, yy, sigma=0.25*k)
    
    # Normalize the filter to have an L1 norm of 1.
    data = data / np.abs(data).sum()

    return data


class PredefinedConvnxn(PredefinedConv):
    def __init__(self, n_channels_in: int, n_channels_out: int, stride: int = 1, k: int = 3, filter_mode: ParameterizedFilterMode = ParameterizedFilterMode.All, n_angles: int = 4, requires_grad=False) -> None:
        super().__init__(n_channels_in, n_channels_out, stride)

        self.padding = k//2

        # Create the convolution kernels with the pre-defined filters.
        w = []
        if filter_mode == ParameterizedFilterMode.Uneven or filter_mode == ParameterizedFilterMode.All:
            w = w + [get_parameterized_filter(k, ParameterizedFilterMode.Uneven, phi) for phi in np.linspace(0, 180, n_angles, endpoint=False)]
        if filter_mode == ParameterizedFilterMode.Even or filter_mode == ParameterizedFilterMode.All:
            w = w + [get_parameterized_filter(k, ParameterizedFilterMode.Even, phi) for phi in np.linspace(0, 180, n_angles, endpoint=False)]
        w = [sign*item for item in w for sign in [-1, 1]]
        
        if filter_mode == ParameterizedFilterMode.Random:
            w = w + [np.random.rand(k, k) * 2. - 1. for _ in range(n_angles)]

        self.n_kernels = len(w)
        w = torch.FloatTensor(np.array(w))
        w = w.unsqueeze(1)
        self.weight = nn.Parameter(w, requires_grad)
        self.filter_requires_grad = requires_grad

        # Create the internal weigths, that have the correct shape for the convolution operation in the forward pass.
        # Note, that the internal weights have to be recumputed after each update of self.weight.
        n_channels_per_kernel = self.n_channels_out // self.n_kernels
        internal_weight = self.weight.data.repeat((n_channels_per_kernel, 1, 1, 1)) 
        self.internal_weight = nn.Parameter(internal_weight, False)


class PredefinedFilterModule(nn.Module):
    """
    The Pre-defined Filter Module (PFM).
    The PFM consists of a dephwise convolution, where the n x n convolution part uses a small set of pre-defined filters. 
    """
    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        filter_mode: ParameterizedFilterMode,
        n_angles: int,
        conv: module = nn.Conv2d,
        f: int = 1,
        k: int = 3,
        stride: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        predefined_filters_require_grad: bool = False,
    ) -> None:
        super().__init__()

        self.predev_conv = PredefinedConvnxn(n_channels_in, n_channels_in * f, stride=stride, k=k, filter_mode=filter_mode, n_angles=n_angles, requires_grad=predefined_filters_require_grad)
        n_channels_mid = self.predev_conv.n_channels_out
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm1 = norm_layer(n_channels_mid)
        self.relu = activation_layer()
        self.conv1x1 = conv(n_channels_mid, n_channels_out, kernel_size=1, stride=1, bias=False)
        self.norm2 = norm_layer(n_channels_out)


    def forward(self, x: Tensor) -> Tensor:
        x = self.predev_conv(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv1x1(x)
        x = self.norm2(x)
        return x


class SmoothConv(nn.Module):
    """
    The SmoothConv module applies Gaussian smoothing to the input feature maps. 
    """
    def __init__(
        self,
        k: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.padding = k//2
        self.stride = stride

        w = [get_parameterized_filter(k, ParameterizedFilterMode.Smooth)]
        w = torch.FloatTensor(np.array(w))
        w = w.unsqueeze(1)
        self.w = nn.Parameter(w, False)


    def forward(self, x: Tensor) -> Tensor:
        n_channels_in = x.shape[1]
        w_tmp = self.w.repeat((n_channels_in, 1, 1, 1))
        out = F.conv2d(x, w_tmp, None, self.stride, self.padding, groups=n_channels_in)
        return out


class DoublePredefinedFilterModule(nn.Module):
    """
    The DoublePredefinedFilterModule module is similar to the Basic Block of ResNet, but uses Pre-defined Filter Modules (PFMs) instead of standard convolution.
    In the forward pass two PFMs are applied. Subsequently, a skip connection is added to the last PFM's output. 
    If a stride of 2 is applied, the skip connection also contains a downsampling layer, where Gaussian smoothing is applied and a subsequent 1 x 1 convolution. 
    """
    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        f: int,
        k: int,
        filter_mode_1: ParameterizedFilterMode,
        filter_mode_2: ParameterizedFilterMode,
        n_angles: int,
        conv: module = nn.Conv2d,
        stride: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer=nn.ReLU,
        downsample: Optional[nn.Module] = nn.Identity(),
        predefined_filters_require_grad: bool = False,
    ) -> None:
        super().__init__()

        self.downsample = downsample
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.onexonedephtwise1 = PredefinedFilterModule(
            n_channels_in,
            n_channels_out,
            f=f,
            k=k,
            filter_mode=filter_mode_1,
            n_angles=n_angles,
            conv=conv,
            stride=stride,
            norm_layer=norm_layer, 
            activation_layer=activation_layer,
            predefined_filters_require_grad=predefined_filters_require_grad,
            )
        self.relu1 = activation_layer(inplace=True)
        self.onexonedephtwise2 = PredefinedFilterModule(
            n_channels_out,
            n_channels_out,
            f=f,
            k=k,
            filter_mode=filter_mode_2,
            n_angles=n_angles,
            conv=conv,
            stride=1,
            norm_layer=norm_layer, 
            activation_layer=activation_layer,
            predefined_filters_require_grad=predefined_filters_require_grad,
            )
        self.relu2 = activation_layer(inplace=True)
        

    def forward(self, x: Tensor) -> Tensor:
        out = self.onexonedephtwise1(x)
        out = self.relu1(out)
        out = self.onexonedephtwise2(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class PFNet_(nn.Module):
    """
    Implementation of the Pre-defined Filter Network (PFNet)
    """
    def __init__(
        self,
        n_classes: int = 102,
        start_config: dict = {
            'k' : 3, 
            'filter_mode' : 'All',
            'n_angles' : 4,
            'n_channels_in' : 3,
            'n_channels_out' : 64,
            'stride' : 2,
            'f' : 16,
            'predefined_filters_require_grad' : False,
        },
        blockconfig_list: List[Tuple[str]] = [
            {'k' : 3, 
            'filter_mode_1' : 'All',
            'filter_mode_2' : 'All',
            'n_angles' : 4,
            'n_blocks' : 2,
            'n_channels_in' : max(64, 64 * 2**(i-1)),
            'n_channels_out' : 64 * 2**i,
            'stride' : 2 if i>0 else 1,
            'f' : 1,
            'predefined_filters_require_grad' : False,
            } for i in range(4)],
        activation: str = 'relu',
        init_mode: str = 'kaiming_normal',
    ) -> None:
        super().__init__()

        if activation == 'relu':
            activation_layer = nn.ReLU
        elif activation == 'leaky_relu':
            activation_layer = nn.LeakyReLU
        self._activation_layer = activation_layer

        self._norm_layer = nn.BatchNorm2d

        self.conv1 = PredefinedFilterModule(
            n_channels_in=start_config['n_channels_in'], 
            n_channels_out=start_config['n_channels_out'],
            stride=start_config['stride'], 
            f=start_config['f'],
            k=start_config['k'],
            filter_mode=ParameterizedFilterMode[start_config['filter_mode']], 
            n_angles=start_config['n_angles'],
            norm_layer=self._norm_layer, 
            activation_layer=self._activation_layer,
            predefined_filters_require_grad=start_config['predefined_filters_require_grad'],
        )
        self.relu1 = activation_layer(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList([
            self._make_layer(
                filter_mode_1=ParameterizedFilterMode[config['filter_mode_1']],
                filter_mode_2=ParameterizedFilterMode[config['filter_mode_2']],
                n_angles=config['n_angles'],
                n_blocks=config['n_blocks'],
                n_channels_in=config['n_channels_in'],
                n_channels_out=config['n_channels_out'],
                stride=config['stride'],
                f=config['f'],
                predefined_filters_require_grad=config['predefined_filters_require_grad'],
            ) for config in blockconfig_list])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(blockconfig_list[-1]['n_channels_out'], n_classes)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, PredefinedConv)) and hasattr(m, 'kernel_size') and m.kernel_size==1:
                if init_mode == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
                elif init_mode == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity=activation)
                elif init_mode == 'sparse':
                    nn.init.sparse_(m.weight, sparsity=0.1, std=0.01)
                elif init_mode == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(
        self,
        filter_mode_1: ParameterizedFilterMode, 
        filter_mode_2: ParameterizedFilterMode,
        n_angles: int,
        n_blocks: int,
        n_channels_in: int,
        n_channels_out: int,
        f: int,
        k: int = 3,
        stride: int = 2,
        predefined_filters_require_grad: bool = False,
    ) -> nn.Sequential:

        norm_layer = self._norm_layer
        activation_layer = self._activation_layer
        
        if stride > 1: 
            skip_smoothing_module = SmoothConv(3)
        else:
            skip_smoothing_module = nn.Identity()

        downsample = nn.Identity()
        if stride > 1 or n_channels_in != n_channels_out:
            conv = nn.Conv2d(n_channels_in, n_channels_out, 1, stride=stride, padding=0, bias=False)
            downsample = nn.Sequential(
                skip_smoothing_module,
                conv,
                norm_layer(n_channels_out),
            )
        
        layers = []

        layers.append(DoublePredefinedFilterModule(
            n_channels_in=n_channels_in,
            n_channels_out=n_channels_out, 
            f=f,
            k=k,
            filter_mode_1=filter_mode_1,
            filter_mode_2=filter_mode_2,
            n_angles=n_angles,
            stride=stride,
            norm_layer=norm_layer, 
            activation_layer=activation_layer, 
            downsample=downsample,
            predefined_filters_require_grad=predefined_filters_require_grad,
        ))

        for _ in range(1, n_blocks):
            layers.append(DoublePredefinedFilterModule(
                n_channels_in=n_channels_out,
                n_channels_out=n_channels_out, 
                f=f,
                k=k,
                filter_mode_1=filter_mode_1,
                filter_mode_2=filter_mode_2,
                n_angles=n_angles,
                stride=1,
                norm_layer=norm_layer, 
                activation_layer=activation_layer,
                predefined_filters_require_grad=predefined_filters_require_grad,
            ))

        return nn.Sequential(*layers)
        

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x