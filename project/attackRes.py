import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models

from utils import compute_gradient, read_image, to_array  # noqa: F401
from transformers import ResNetForImageClassification


def func(inp, net=None, target=None):
    """Compute negative log likelihood
    Parameters
    ----------
    inp : torch.Tensor
        Input image (single image batch)
    net: torch.nn.module
        Classifier network
    target : int
        Imagenet ground truth label id.
    Returns
    -------
    loss : torch.Tensor
        Loss for the 'inp' image.
    """
    out = net(inp)
    loss = torch.nn.functional.nll_loss(out, target=torch.LongTensor([target]))
    print(f"Loss: {loss.item()}")
    return loss


def attack(tensor, net, eps=.001, n_iter=50):
    """Run the fast sign gradient method attack
    Parameters
    ----------
    tensor : torch.Tensor
        The input image of shape '(1,3,224,224)'
    net : torch.nn.Module
        Classifier network.
    eps : float
        Determines how much to modify in an interatin
    n_iter : int
        number of iterations
    Returns
    -------
    new_tensor : torch.Tensor
        New image that is a slight modification but will
        "fool" the classifier
    """
    new_tensor = tensor.detach().clone()
    orig_prediction = net(tensor).argmax()

    for _ in range(n_iter):
        net.zero_grad()
        grad = compute_gradient(
            func, new_tensor, net=net,
            target=orig_prediction.item())
        new_tensor = torch.clamp(new_tensor + eps * grad.sign(), -2, 2)
        new_prediction = net(new_tensor).argmax()
        if orig_prediction != new_prediction:
            print(f"new new_prediction : {orig_prediction.item()}")
            print(f"new new_prediction : {new_prediction.item()}")
            break
    return new_tensor, orig_prediction.item(), new_prediction.item()


if __name__ == "__main__":

    model = models.resnet50(pretrained=True)
    net = model
    net.eval()

    tensor = read_image("photos/jacamar.jpg")

    new_tensor, orig_prediction, new_prediction = attack(
        tensor, net, eps=1e-3, n_iter=100
    )
    print(orig_prediction, new_prediction)

    _, (ax_orig, ax_new, ax_diff) = plt.subplots(1, 3, figsize=(19.2, 10.8))
    # _, (ax_new) = plt.subplots(1, 1)
    arr = to_array(tensor)
    new_arr = to_array(new_tensor)
    diff_arr = np.abs(arr-new_arr).mean(axis=-1)
    diff_arr = diff_arr / diff_arr.max()

    ax_orig.imshow(arr)
    ax_new.imshow(new_arr)
    ax_diff.imshow(diff_arr, cmap="gray")

    ax_orig.axis("off")
    ax_new.axis("off")
    ax_diff.axis("off")

    ax_orig.set_title("original: jacamar")
    ax_new.set_title("Modfiied: feather boa")
    ax_diff.set_title("Difference")

    plt.savefig("photos/bird.png")
