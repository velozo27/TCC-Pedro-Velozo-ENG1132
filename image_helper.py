import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np


class ImageHelper:
    """
    A helper class for working with image data in PyTorch.
    """

    def __init__(self):
        """
        Initializes a new instance of the ImageHelper class.
        """
        return
    
    def image_to_tensor(self, image: Image or str) -> torch.Tensor:
        if type(image) == str:
            image = Image.open(image)

        transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        return transform(image)

    def show_tensor_as_image(self, tensor: torch.Tensor):
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


    def show_tensors_side_by_side(
        tensors: list[dict[str, torch.Tensor]],
    ) -> None:
        """
        Displays a list of images represented as PyTorch tensors side by side.

        Args:
            tensors: A list of PyTorch tensors representing the images. Each tensor should have shape (C, H, W),
                where C is the number of channels, H is the height, and W is the width.

        Returns:
            None
        """
        num_tensors = len(tensors)
        fig, axes = plt.subplots(nrows=1, ncols=num_tensors)

        for i in range(num_tensors):
            tensor = tensors[i]
            tensor_np = tensor.numpy()
            axes[i].imshow(tensor_np.transpose((1, 2, 0)))

        plt.show()

    def open_image(self, image_path: str) -> Image:
        return Image.open(image_path)

    def open_and_show_image(self, image_path: str) -> None:
        image = Image.open(image_path)
        # plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.show()

    def downsample_image(
        self,
        image: Image, downsample_factor: int, resample_filter: int
    ) -> Image:
        width, height = image.size
        new_width = width // downsample_factor
        new_height = height // downsample_factor
        return image.resize((new_width, new_height), resample=resample_filter)

    def downsample_image_as_tensor(
        self,
        image: Image or str, downsample_factor: int, interpolation=Image.LINEAR
    ) -> torch.Tensor:
        if type(image) == str:
            image = Image.open(image)

        width, height = image.size
        new_width = width // downsample_factor
        new_height = height // downsample_factor
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((new_height, new_width), interpolation=interpolation),
            ])
        return transform(image)
    
    def downsample_image_as_tensor_and_show(
        self,
        image: Image or str, downsample_factor: int, interpolation=Image.LINEAR
    ) -> None:
        tensor = self.downsample_image_as_tensor(image, downsample_factor, interpolation)
        self.show_tensor_as_image(tensor)
    
    def downsample_and_upsample_image_as_tensor(
        self,
        image: Image or str, downsample_factor: int, interpolation=Image.BICUBIC
    ) -> torch.Tensor:
        if type(image) == str:
            image = Image.open(image)

        width, height = image.size
        new_width = width // downsample_factor
        new_height = height // downsample_factor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((new_height, new_width), interpolation=interpolation),
            transforms.Resize((height, width), interpolation=interpolation),
        ])
        return transform(image)
    
    def downsample_and_upsample_image_as_tensor_and_show(
        self,
        image: Image or str, downsample_factor: int, interpolation=Image.BICUBIC
    ) -> None:
        if type(image) == str:
            image = Image.open(image)
        tensor = self.downsample_and_upsample_image_as_tensor(image, downsample_factor, interpolation)
        self.show_tensor_as_image(tensor)
    
    def apply_model_to_image(
        self,
        model: torch.nn.Module,
        image: Image or str,
        downsample_factor: int,
    ) -> torch.Tensor:
        if type(image) == str:
            image = Image.open(image)

        tensor = self.downsample_and_upsample_image_as_tensor(image, downsample_factor)

        return model(tensor)

    def apply_model_to_image_and_show(
        self,
        model: torch.nn.Module,
        image: Image or str,
        downsample_factor: int,
    ) -> None:
        tensor = self.apply_model_to_image(model, image, downsample_factor)
        self.show_tensor_as_image(tensor)

    def show_tensor_as_images_side_by_side(self, tensors: list[dict[str, torch.Tensor]]) -> None:
        num_tensors = len(tensors)
        fig, axes = plt.subplots(nrows=1, ncols=num_tensors, figsize=(14, 8))

        for index, tensor_dict in enumerate(tensors):
            tensor = tensor_dict['tensor']
            label = tensor_dict['label']
        
            try: 
                tensor_np = tensor.numpy()
            except:
                tensor_np = tensor.detach().numpy()

            axes[index].imshow(tensor_np.transpose((1, 2, 0)), aspect='auto')
            axes[index].set_title(label)

        plt.show()

    def get_differance_between_tensors(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        return torch.abs(tensor1 - tensor2)

    def get_differance_between_image_and_show(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        differance = self.get_differance_between_tensors(tensor1, tensor2)
        return differance

    