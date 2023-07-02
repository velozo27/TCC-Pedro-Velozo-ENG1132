import os
from torchvision.io import read_image
from patchify import patchify
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2

FREQUENCY_THRESHOLD = 2500

def create_valid_image_patches(img_dir: str, patch_size, output_path: str, num_patches: int, frequency_threshold=FREQUENCY_THRESHOLD) -> None:
    """
    Creates patches from valid images in a given directory and saves them to a new directory.

    Args:
        img_dir (str): The path to the directory containing the images.
        patch_size (tuple): The desired size of each patch.
        output_path (str): The path to the output directory to save the patches.
        num_patches (int): The number of patches to extract from each image.
        frequency_threshold (int): The threshold for the mean frequency magnitude. Defaults to FREQUENCY_THRESHOLD.

    Returns:
        None
    """
    # Create a new directory to store the patches
    patch_dir = os.path.join(output_path, "patches")
    os.makedirs(patch_dir, exist_ok=True)

    # Loop through each image in the directory
    files = os.listdir(img_dir)
    files.sort()
    for img_name in tqdm(files):
        try:
          # if the file is not an image then skip to next iteration
          if os.path.isdir(img_name):
            continue 

          # Load the image
          img_path = os.path.join(img_dir, img_name)
          img = np.array(Image.open(img_path))

          # Extract patches from the image using patchify
          patches = patchify(img, patch_size, step=patch_size[0])

          # Shuffle the patches !!!
          patches = patches.reshape(-1, patch_size[0], patch_size[1], patch_size[2])
          patches = patches[np.random.permutation(patches.shape[0])]

          # Save the patches to the new directory
          img_patch_dir = os.path.join(patch_dir, f"{os.path.splitext(img_name)[0]}_patch")
          os.makedirs(img_patch_dir, exist_ok=True)

          success_count = 0
          error_count = 0
          for i in range(min(patches.shape[0], num_patches)):
            patch = patches[i]
            patch = Image.fromarray(patch)
            is_valid = is_image_valid(patch.convert('L'), frequency_threshold)
            if not is_valid:
               error_count += 1
               continue

            save_path = os.path.join(img_patch_dir, f"patch_{i}.png")

            patch.save(save_path)

            success_count += 1

        except IsADirectoryError:
          continue

def is_image_valid(image: Image, frequency_threshold=FREQUENCY_THRESHOLD) -> bool:
  """
  Checks if an image is valid based on its frequency content.

  Args:
      image (Image): The input image to validate.
      frequency_threshold (int): The threshold for the mean frequency magnitude. Defaults to 2000.

  Returns:
      bool: True if the image is valid, False otherwise.
  """

  image = image.convert('L')  # convert image to grayscale

  image_np = np.array(image)

  # calculating the discrete Fourier transform
  DFT = cv2.dft(np.float32(image_np), flags=cv2.DFT_COMPLEX_OUTPUT)

  # reposition the zero-frequency component to the spectrum's middle
  shift = np.fft.fftshift(DFT)
  row, col = image_np.shape
  center_row, center_col = row // 2, col // 2

  # create a mask with a centered square of 1s
  mask = np.zeros((row, col, 2), np.uint8)
  mask[center_row - 30:center_row + 30, center_col - 30:center_col + 30] = 1

  # put the mask and inverse DFT in place.
  fft_shift = shift * mask
  fft_ifft_shift = np.fft.ifftshift(fft_shift)
  imageThen = cv2.idft(fft_ifft_shift)

  # calculate the magnitude of the inverse DFT
  imageThen = cv2.magnitude(imageThen[:, :, 0], imageThen[:, :, 1])

  np_imageThen = np.array(imageThen)
  np.mean(np_imageThen)

  return np.mean(np_imageThen) > frequency_threshold

def create_image_patches(img_dir: str, patch_size, output_path: str, num_patches: int) -> None:
    """
    Creates patches from images in a given directory and saves them to a new directory.

    Args:
        img_dir (str): The path to the directory containing the images.
        patch_size (tuple): The desired size of each patch.
        num_patches (int): The number of patches to extract from each image.
    """
    # Create a new directory to store the patches
    patch_dir = os.path.join(output_path, "patches")
    os.makedirs(patch_dir, exist_ok=True)

    # Loop through each image in the directory
    files = os.listdir(img_dir)
    files.sort()
    for img_name in tqdm(files):
        try:
          # if the file is not an image then skip to next iteration
          if os.path.isdir(img_name):
            continue 

          # Load the image
          img_path = os.path.join(img_dir, img_name)
          img = np.array(Image.open(img_path))

          # Extract patches from the image using patchify
          patches = patchify(img, patch_size, step=patch_size[0])

          # Shuffle the patches !!!
          patches = patches.reshape(-1, patch_size[0], patch_size[1], patch_size[2])
          patches = patches[np.random.permutation(patches.shape[0])]

          # Save the patches to the new directory
          img_patch_dir = os.path.join(patch_dir, f"{os.path.splitext(img_name)[0]}_patch")
          os.makedirs(img_patch_dir, exist_ok=True)

          for i in range(min(patches.shape[0], num_patches)):
            patch = patches[i]
            patch = Image.fromarray(patch)
            save_path = os.path.join(img_patch_dir, f"patch_{i}.png")

            patch.save(save_path)

        except IsADirectoryError:
          continue

def read_image_aux(file_path):
    """
    Reads and loads an image file from the specified path.

    Args:
        file_path (str): The path to the image file.

    Returns:
        PIL.Image.Image: The loaded image.
    """
    try:
        image = Image.open(file_path)
        return image
    except (IOError, OSError) as e:
        # Handle empty or corrupted files
        raise ValueError(f"Error loading image file: {file_path}. Reason: {str(e)}")


def get_image_patch(img_dir, img_index=-1, patch_index=-1):
    """
    Gets an image from a dataset.

    Args:
        dataset (Dataset): The dataset to get the image from.
        index (int): The index of the image to get.

    Returns:
        Tensor: The image.
    """
    imgs_dir = os.path.join(img_dir, "patches")
    print(imgs_dir)
    imgs = list(os.listdir(imgs_dir))
    if img_index == -1:
      img_index = np.random.randint(0, len(imgs))

    patches_dir = os.path.join(imgs_dir, imgs[img_index])
    print(patches_dir)

    patches = list(os.listdir(patches_dir))
    if patch_index == -1:
      patch_index = np.random.randint(0, len(patches))
    return read_image(os.path.join(patches_dir, patches[patch_index]))

        