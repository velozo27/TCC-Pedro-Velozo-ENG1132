import matplotlib.pyplot as plt
import torch


def show_tensor_as_image(tensor: torch.Tensor):
    """
    Displays an image represented as a PyTorch tensor.

    Args:
        tensor: A PyTorch tensor representing the image. The tensor should have shape (C, H, W),
            where C is the number of channels, H is the height, and W is the width.

    Returns:
        None
    """
    plt.figure()

    if torch.is_tensor(tensor):
        tensor_np = tensor.detach().numpy()
    else:
        tensor_np = tensor.numpy()

    plt.imshow(tensor_np.transpose((1, 2, 0)))
    plt.show()


def tensor_as_image(tensor: torch.Tensor):
    """
    Converts a tensor to an image.

    Args:
        tensor: A PyTorch tensor representing the image. The tensor should have shape (C, H, W),
            where C is the number of channels, H is the height, and W is the width.

    Returns:
        The image as a numpy array.
    """
    if torch.is_tensor(tensor):
        tensor_np = tensor.detach().numpy()
    else:
        tensor_np = tensor.numpy()
    return tensor_np.transpose((1, 2, 0))

    

