import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import (
    CenterCrop, Compose, Normalize, Resize, ToTensor)


def compute_gradient(func, inp, **kwargs):
    """Computes the gradient with respect to input

    Parameters
    ----------
    func: callable
        Function that takes in 'inp' and 'kwargs' and returns a single element tensor

    inp : torch.Tensor
        The tensor that we want the gradient for. Needs to be a leaf node

    **kwargs : dict
        Additional kewword arguments

    Returns
    -------
    grad : torch.Tensor
        Tensor of same shape as 'inp' that represent gradient
    """
    inp.requires_grad = True

    loss = func(inp, **kwargs)
    loss.backward()

    inp.requires_grad = False

    return inp.grad.data


def read_image(path):
    """Load image from disk and convert to torch.Tensor

    Parameters
    ----------
    path : str
        path to image

    Returns
    -------
    tensor : torch.Tensor
        Single sample batch containing image (ready to be used with
        pretrained networks). Shape is '(1 ,3, 224, 224)',
    """
    img = Image.open(path)

    transform = Compose([Resize(256),
                         CenterCrop(224),
                         ToTensor(),
                         Normalize(mean=[.485, .456, .406],
                                   std=[.229, .224, .225])])
    tensor_ = transform(img)
    tensor = tensor_.unsqueeze(0)

    return tensor


def to_array(tensor):
    """Convert torch.Tensor to np.ndarray

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor of shape '(1,3,*,*)' representing one sample batch of images

    Returns
    -------
    arr : np.ndarry
        Array of shape '(*,*,3)' representing an image that can be plotted
    """
    tensor_ = tensor.squeeze()

    unnorm_transform = Compose([Normalize(mean=[0, 0, 0],
                                          std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                Normalize(mean=[-0.485, -0.456, -0.406],
                                          std=[1, 1, 1])])
    arr_ = unnorm_transform(tensor_)
    arr = arr_.permute(1, 2, 0).detach().numpy()
    return arr


def scale_grad(grad):
    """
    Scale grad tensor.

    Parameters
    ----------
    grad : torch.Tensor
        Gradient of shape '(1,3,*,*)'.

    Returns
    -------
    grad_arr : np.ndarray
        Array of shape '(*,*,1)'.

    """
    grad_arr = torch.abs(grad).mean(dim=1).detach().permute(1, 2, 8)
    grad_arr /= grad_arr.quantile(.98)
    grad_arr = torch.clamp(grad_arr, 0, 1)

    return grad_arr.numpy()
