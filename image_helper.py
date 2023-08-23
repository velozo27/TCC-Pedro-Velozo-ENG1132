import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np
from torchvision.utils import save_image


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
            tensor_np = tensor.cpu().detach().numpy()
        else:
            tensor_np = tensor.numpy()

        try:
            plt.imshow(tensor_np.transpose((1, 2, 0)))
            plt.show()
        except:
            tensor_np = np.squeeze(tensor_np, axis=0)
            plt.imshow(tensor_np.transpose((1, 2, 0)))
            plt.show()


    def show_tensors_side_by_side(
        tensors,
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
        image: Image or str, downsample_factor: int, interpolation=Image.BICUBIC,
        unsqueeze: bool = False,
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
        
        if unsqueeze:
            return transform(image).unsqueeze(0)
        
        return transform(image)
    
    def downsample_image_as_tensor_and_show(
        self,
        image: Image or str, downsample_factor: int, interpolation='LINEAR'
    ) -> None:
        tensor = self.downsample_image_as_tensor(image, downsample_factor, interpolation)
        self.show_tensor_as_image(tensor)
    
    def save_tensor_as_image(self, tensor: torch.Tensor, save_path: str) -> None:
        img = tensor

        if not save_path.endswith('.png'):
            save_path += '.png'

        save_image(img, save_path)

    def downsample_image_as_tensor_and_show_and_save(
        self,
        image: Image or str, downsample_factor: int, interpolation=Image.BICUBIC, save_path: str = None
    ) -> None:
        tensor = self.downsample_image_as_tensor(image, downsample_factor, interpolation)
        self.show_tensor_as_image(tensor)

        if save_path:
            self.save_tensor_as_image(tensor, save_path)
    
    def downsample_and_upsample_image_as_tensor(
        self,
        image: Image or str, downsample_factor: int, interpolation=Image.BICUBIC,
        unsqueeze: bool = False,
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

        if unsqueeze:
            return transform(image).unsqueeze(0)

        return transform(image)
    
    def downsample_and_upsample_image_as_tensor_and_show(
        self,
        image: Image or str, downsample_factor: int, interpolation=Image.BICUBIC
    ) -> None:
        if type(image) == str:
            image = Image.open(image)
        tensor = self.downsample_and_upsample_image_as_tensor(image, downsample_factor, interpolation)
        self.show_tensor_as_image(tensor)

    def reshape_image_to_dimensions_to_tensor(
        self,
        image: Image or str, new_width: int, new_height: int, interpolation=Image.BICUBIC,
        unsqueeze: bool = False,
    ) -> torch.Tensor:
        if type(image) == str:
            image = Image.open(image)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((new_height, new_width), interpolation=interpolation),
        ])

        if unsqueeze:
            return transform(image).unsqueeze(0)

        return transform(image)

    def crop_image_to_nearest_even_dimensions(self, image: Image) -> Image:
        width, height = image.size
        new_width = width
        new_height = height
        if width % 2 != 0:
            new_width = width - 1
        if height % 2 != 0:
            new_height = height - 1

        # new_width = width - (width % 2)
        # new_height = height - (height % 2)
        return transforms.Compose([
            transforms.CenterCrop((new_height, new_width)),
        ])(image)

    def crop_image_to_nearest_even_dimensions_and_transform_to_tensor(self, image: Image) -> Image:
        image = self.crop_image_to_nearest_even_dimensions(image)
        return transforms.Compose([
            transforms.ToTensor(),
        ])(image)
    
    def apply_model_to_image(
        self,
        model: torch.nn.Module,
        image: Image or str,
        downsample_factor: int,
        unsqueeze = False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        should_upsample = True,
    ) -> torch.Tensor:
        if type(image) == str:
            image = Image.open(image)

        # usefull for SRCNN
        if should_upsample:
            tensor = self.downsample_and_upsample_image_as_tensor(image, downsample_factor, unsqueeze=unsqueeze)
        else: 
            tensor = self.downsample_image_as_tensor(image, downsample_factor, unsqueeze=unsqueeze)

        return model(tensor.to(device))

    def apply_model_to_image_and_show(
        self,
        model: torch.nn.Module,
        image: Image or str,
        downsample_factor: int,
        unsqueeze: bool = False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        should_upsample = True,
    ) -> None:
        print('should_upsample MUST ONLY BE `TRUE` for SRCNN')
        tensor = self.apply_model_to_image(model, image, downsample_factor, unsqueeze=unsqueeze, device=device, should_upsample=should_upsample)
        self.show_tensor_as_image(tensor)

    def show_tensor_as_images_side_by_side(self, tensors, show_grid=True) -> None:
        num_tensors = len(tensors)
        fig, axes = plt.subplots(nrows=1, ncols=num_tensors, figsize=(14, 8))

        for index, tensor_dict in enumerate(tensors):
            tensor = tensor_dict['tensor']
            label = tensor_dict['label']
        
            try: 
                tensor_np = tensor.cpu().numpy()
            except:
                tensor_np = tensor.cpu().detach().numpy()

            try:
                axes[index].imshow(tensor_np.transpose((1, 2, 0)), aspect='auto')
            except:
                tensor_np = np.squeeze(tensor_np, axis=0)
                axes[index].imshow(tensor_np.transpose((1, 2, 0)), aspect='auto') 

            axes[index].set_title(label)

            if not show_grid:
                axes[index].axis('off') 
 
        plt.show()

    def show_tensors_custom_grid(tensors, rows=2, cols=2) -> None:
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))

        for index, tensor_dict in enumerate(tensors):
            row = index // cols
            col = index % cols

            tensor = tensor_dict['tensor']
            label = tensor_dict['label']

            try:
                tensor_np = tensor.cpu().numpy()
            except:
                tensor_np = tensor.cpu().detach().numpy()

            try:
                axes[row, col].imshow(tensor_np.transpose((1, 2, 0)), aspect='auto')
            except:
                tensor_np = np.squeeze(tensor_np, axis=0)
                axes[row, col].imshow(tensor_np.transpose((1, 2, 0)), aspect='auto')

            axes[row, col].set_title(label)

            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

    def get_differance_between_tensors(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        return torch.abs(tensor1 - tensor2)

    def get_differance_between_image_and_show(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        differance = self.get_differance_between_tensors(tensor1, tensor2)
        return differance

    