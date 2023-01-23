# Pre-defined Filter Networks

Pre-defined Filter Convolutional Neural Networks (PFCNNs) are CNNs where the n x n convolution kernels with n>1 are pre-defined and constant during training.
Training occurs end-to-end by simply finding linear combinations of the outputs of the pre-defined filters using gradient descent.
This repository implements a PFCNN as a residual network with depth-wise convolution. In the n x n convolution operations with n > 1 the filters can be freely chosen by the user.
The implementation currently focuses on edge filters with different orientations.

# Get started:
Clone this repository, either by downloading the files in zipped format directly from Github or using:

    git clone https://github.com/Criscraft/HackathonDigitaltag2022.git

Go to the folder where you have downloaded/cloned the Git

    cd PredefinedFilterNetworks

We recommend to use a virtual environment. There are many options how to setup virtual environments. Please have a look at [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) or [venv](https://docs.python.org/3/library/venv.html).
For example if you wish to work with conda you can create a new Python environment using:

    conda create --name myenv python=3.8 pip

If you prefer, you can give any desired name for the created enviorment by replacing myenv with your desired name.
Subsequently, activate the enviorment with:

    conda activate myenv

Install the requirements from the requirements.txt file:

    pip install -r requirements.txt

Please test, if you have GPU support. Open an interactive python session

    python

and type

    import torch
    print(torch.cuda.is_avaliable())

The output should be "True".
If not, please check if your [CUDA](https://developer.nvidia.com/cuda-downloads) version is up-to-date. Then, go to the [Pytorch website](https://pytorch.org/), insert your specifications there. You will receive an install command for pytorch that suits your CUDA version. After installing the Pytorch version that suits your setup you should have GPU support.

# Get started:
The repository contains a script "example.py" which shows how to load the HFNet model.
You can run the script using:

    python example.py

# Requirements

Tested for Python 3.8.0

    numpy==1.23.2
    Pillow==9.2.0
    torch==1.12.1+cu116
    torchinfo==1.7.0
    torchvision==0.13.1+cu116