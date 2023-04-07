import matplotlib.pyplot as plt
import torch


class ImageHelper():
    def __init__(self) -> None:
        return

    def show_tensor_as_image(tensor: torch.Tensor) -> None:
        plt.figure()

        tensor_np = tensor.numpy()
        plt.imshow(tensor_np.transpose((1, 2, 0)))

        plt.show()
