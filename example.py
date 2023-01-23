import torch
from torchinfo import summary
from models.PFNet import PFNet


def summarize_network(network, input_size):
    device = torch.device("cpu")
    network = network.to(device)
    return summary(network, input_size)

hfnet18 = PFNet(
    n_classes = 100,
    start_config = {
        'k' : 3, 
        'filter_mode' : 'All',
        'n_angles' : 4,
        'n_channels_in' : 3,
        'n_channels_out' : 64,
        'stride' : 2,
        'f' : 16,
        'handcrafted_filters_require_grad' : False,
    },
    blockconfig_list = [
        {'k' : 3, 
        'filter_mode_1' : 'All',
        'filter_mode_2' : 'All',
        'n_angles' : 4,
        'n_blocks' : 2,
        'n_channels_in' : max(64, 64 * 2**(i-1)),
        'n_channels_out' : 64 * 2**i,
        'stride' : 2 if i>0 else 1,
        'f' : 1,
        'handcrafted_filters_require_grad' : False,
        } for i in range(4)],
)

summarize_network(hfnet18, (1, 3, 224, 224))