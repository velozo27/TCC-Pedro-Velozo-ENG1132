import time
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
from torchmetrics import PeakSignalNoiseRatio
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from SRCNN import SRCNN
from loops import train_loop, validation_loop
from torchmetrics.image import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
from image_helper import ImageHelper


class ModelRunner():
    def __init__(self, model=None, device="cuda" if torch.cuda.is_available() else "cpu", train_loop=train_loop, validation_loop=validation_loop):
        self.device = device

        # not used for now
        self.model = model

        self.train_loop = train_loop
        self.validation_loop = validation_loop

        self.epoch_array = []
        self.time_array = []
        self.lr_array = []
        self.train_loss_array = []
        self.validation_loss_array = []

        self.model_df = None

    def clear_all(self, clear_torch_cache=True) -> None:
        self.clear_model_df()
        self.clear_metrics()
        if clear_torch_cache:
            self.clear_torch()

    def clear_torch(self) -> None:
        torch.cuda.empty_cache()

    def clear_model_df(self) -> None:
        self.model_df = None

    def clear_metrics(self) -> None:
        self.epoch_array.clear()
        self.time_array.clear()
        self.lr_array.clear()
        self.train_loss_array.clear()
        self.validation_loss_array.clear()

    def get_metrics(self):
        return self.epoch_array, self.time_array, self.lr_array, self.train_loss_array, self.validation_loss_array

    # TODO: rever se está funcionando
    def load_model(self, model: nn.Module, model_weights_path: str) -> torch.nn.Module:
        try:
            model.load_state_dict(torch.load(model_weights_path))
        except:
            model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float, path: str) -> None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)

    def load_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, path: str) -> None:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return model, optimizer, epoch, loss

    def load_essential_data_from_pth_and_csv(self,
                                             model: nn.Module,
                                             pth_path: str, csv_path: str) -> (nn.Module, pd.DataFrame):
        model = self.load_model(model, pth_path)
        df = self.load_df(csv_path)

        return model, df

    def save_model_weights(self, model: nn.Module, model_weights_path: str) -> None:
        torch.save(model.state_dict(), model_weights_path)

    def create_train_loss_plot_from_df_path(self, model_df_path: str, plot_path: str) -> None:
        pass

    def plot_time_per_epoch_comparision(self,
                                        dfs
                                        ) -> None:
        fig = plt.figure(figsize=(10, 10))
        for df in dfs:
            plt.plot(df['epoch'], df['time'])
        plt.title('Time per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Time (s)')
        plt.legend()
        plt.show()

    def plot_time_per_epoch_comparision(self, dfs_data) -> None:
        fig = plt.figure(figsize=(10, 10))
        for data in dfs_data:
            label = data['label']
            df = data['df']
            plt.plot(df['epoch'], df['time'], label=label)
        plt.title('Time per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Time (s)')
        plt.legend()
        plt.show()

    def plot_time_per_epoch_from_df(self, df: pd.DataFrame) -> None:
        fig = plt.figure(figsize=(10, 10))
        plt.plot(df['epoch'], df['time'])
        plt.title('Time per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Time (s)')
        plt.show()

    def plot_train_validation_loss_from_df(self, df: pd.DataFrame, show_lr=True) -> None:
        if df is None:
            df = self.get_model_df()

        fig = plt.figure(figsize=(10, 10))
        plt.plot(df['epoch'], df['train_loss'], label='train_loss')
        plt.plot(df['epoch'], df['validation_loss'], label='validation_loss')

        if show_lr:
            plt.plot(df['epoch'], df['lr'], label='lr')
        plt.title('Train and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_lr_comparison(self, dfs) -> None:
        fig = plt.figure(figsize=(10, 10))
        for df_dict in dfs:
            df = df_dict["df"]
            label = df_dict["label"]
            plt.plot(df['epoch'], df['lr'], label=f'{label} lr')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.show()

    def plot_train_validation_loss_comparision(self,
                                               dfs,
                                               show_lr=True
                                               ) -> None:
        fig = plt.figure(figsize=(10, 10))
        for df_dict in dfs:
            df = df_dict["df"]
            label = df_dict["label"]
            plt.plot(df['epoch'], df['train_loss'],
                     label=f'{label} train_loss')
            plt.plot(df['epoch'], df['validation_loss'],
                     label=f'{label} validation_loss')
            if show_lr:
                plt.plot(df['epoch'], df['lr'], label=f'{label} lr')
        plt.title('Train and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def generate_dummy_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "epoch": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "time": [60, 60.6, 61, 60, 60.6, 61, 60, 60.6, 61, 60],
            "lr": [0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0],
            "train_loss": [i for i in range(10, 0, -1)],
            "validation_loss": [i for i in range(10, 0, -1)]
        })

    def load_df(self, df_path: str) -> pd.DataFrame:
        return pd.read_csv(df_path)

    def calculate_psnr(self,
                       img1: torch.Tensor,
                       img2: torch.Tensor) -> float:
        
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)

        psnr = PeakSignalNoiseRatio().to(self.device)
        return psnr(img1.to(self.device), img2.to(self.device))

    def calculate_ssim(self,
                       img1: torch.Tensor,
                       img2: torch.Tensor) -> float:

        # add batch dimension, since ssim expects a batch of images and not a single image. And unsqueeze adds a dimension at the specified position
        if len(img1.shape) == 3:
            img1 = img1.unsqueeze(0)
        if len(img2.shape) == 3:
            img2 = img2.unsqueeze(0)
            
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        return ssim(img1.to(self.device), img2.to(self.device))

    def plot_lr_from_df(self, df=None):
        if df is None:
            df = self.get_model_df()

        fig = plt.figure(figsize=(10, 10))
        plt.plot(df['epoch'], df['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.show()

    def compare_models(self,
                       models,
                       images_path: str,
                       scale=3,
                       show_bicubic=True,
                       ) -> pd.DataFrame:

        pathlist = Path(images_path).rglob('*.png')
        number_of_images = len(list(pathlist))

        if show_bicubic:
            # add to models dict a bicubic "model""
            models.append({
                "name": "Bicubic",
                "model": None,
                "scale": scale
            })

        result_dict = {}  # Dictionary to store both model and bicubic values

        for model_dict in models:
            model_psnr_avg = 0
            model_ssim_avg = 0
            
            model_name = model_dict["name"]
            model = model_dict["model"]
            
            scale = model_dict["scale"] if "scale" in model_dict else scale
            unsqueeze = model_dict["unsqueeze"] if "unsqueeze" in model_dict else False

            pathlist = Path(images_path).rglob('*.png')

            # Initialize values for model and bicubic
            result_dict[model_name] = {
                "PSNR": 0,
                "SSIM": 0,
            }

            for img_path in tqdm(pathlist):
                path_in_str = str(img_path)
                image = Image.open(path_in_str)
                
                # Essa código diminui o valor dos PSNR's para todos os modelos. Mas deixa de quebrar os modelos não pre-upsampled (ex: SRCNN)
                # print()
                # print('image ANTES =', image.size)
                image_helper = ImageHelper()
                # image = image_helper.crop_image_to_nearest_even_dimensions(image)
                # print('image DEPOIS =', image.size)

                if "input_transform" in model_dict:
                     # custom transform
                     input_transform = model_dict["input_transform"]
                else:
                    # default transform for SRCNN (bicubic downsample and upsample)
                    def input_transform(input):
                        return transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize(
                                (input.size[1] // scale, input.size[0] // scale), interpolation=Image.BICUBIC),
                            transforms.Resize(
                                (input.size[1], input.size[0]), interpolation=Image.BICUBIC)
                        ])(input)

                input_image = input_transform(image)

                # some models return a 3D tensor, so we need to unsqueeze it to 4D
                if unsqueeze:
                    input_image = input_image.unsqueeze(0)
                    
                targets = transforms.ToTensor()(image).to(self.device)

                # targets = image_helper.crop_image_to_nearest_even_dimensions_and_transform_to_tensor(image).to(self.device)

                if model_name == "Bicubic":
                    # bicubic preditcion (we only apply the default transform to the input image) 
                    preds = input_image.to(self.device)
                else:
                    # model prediction
                    with torch.no_grad():
                        model_image = model(input_image.to(self.device))
                    # print('model_image =', model_image.shape)
                    preds = model_image.to(self.device)

                # if model is not None:
                if model_name is "Bicubic":
                    pass 
                elif len(preds.shape) == 4:
                    targets = image_helper.reshape_image_to_dimensions_to_tensor(image, model_image.shape[3], model_image.shape[2]).to(self.device)
                else:
                    targets = image_helper.reshape_image_to_dimensions_to_tensor(image, model_image.shape[2], model_image.shape[1]).to(self.device)

                model_psnr_avg += self.calculate_psnr(preds, targets)    
                model_ssim_avg += self.calculate_ssim(preds, targets)
    
            model_psnr_avg /= number_of_images    
            model_ssim_avg /= number_of_images

            result_dict[model_name]["PSNR"] += model_psnr_avg.item()
            result_dict[model_name]["SSIM"] += model_ssim_avg.item()

        df = pd.DataFrame.from_dict(result_dict, orient="index")
        return df

    def get_model_df(self) -> pd.DataFrame:
        if self.model_df is None or self.model_df.empty:
            self.model_df = pd.DataFrame({
                "epoch": self.epoch_array,
                "time": self.time_array,
                "lr": self.lr_array,
                "train_loss": self.train_loss_array,
                "validation_loss": self.validation_loss_array
            })
            return self.model_df
        else:
            return self.model_df

    def save_model_df(self, model_df_path: str) -> None:
        df = self.get_model_df()
        df.to_csv(model_df_path, index=False)

    def train(self,
          model: nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          validation_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
          epochs=10,
          loss_fn=nn.MSELoss(),
          save_file_path=None
          ) -> None:

        def save_print_to_file(print_string):
            if save_file_path is not None:
                with open(save_file_path, 'a') as f:
                    f.write(print_string + '\n')
            print(print_string)

        for current_epoch in range(epochs):
            epoch_string = f"\nepoch {current_epoch}\n-------------------------------"
            save_print_to_file(epoch_string)

            start_time = time.time()

            train_loss = train_loop(
                train_dataloader, model, loss_fn, optimizer)
            validation_loss = validation_loop(
                validation_dataloader, model, loss_fn)

            if scheduler is not None:
                lr_before = f"Learning rate (antes): {optimizer.param_groups[0]['lr']}"
                scheduler.step()
                lr_after = f"Learning rate (depois): {optimizer.param_groups[0]['lr']}"
                save_print_to_file(lr_before)
                save_print_to_file(lr_after)

            self.train_loss_array.append(train_loss)
            train_loss_string = f"Train loss: {train_loss}"
            save_print_to_file(train_loss_string)

            self.validation_loss_array.append(validation_loss)
            validation_loss_string = f"Validation loss: {validation_loss}"
            save_print_to_file(validation_loss_string)

            elapsed_time = time.time() - start_time
            time_string = f"Elapsed time: {elapsed_time}"
            save_print_to_file(time_string)

    def get_arrays_from_df(self, df: pd.DataFrame) -> (list, list, list, list, list):
        epoch_array = df['epoch'].to_list()
        time_array = df['time'].to_list()
        lr_array = df['lr'].to_list()
        train_loss_array = df['train_loss'].to_list()
        validation_loss_array = df['validation_loss'].to_list()

        return epoch_array, time_array, lr_array, train_loss_array, validation_loss_array
    
    def train_from_checkpoint(self,
                model: nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                validation_dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = None,
                df: pd.DataFrame = None,
                epoch_array: list = None,
                time_array: list = None,
                lr_array: list = None,
                train_loss_array: list = None,
                validation_loss_array: list = None,
                start_epoch: int = None,
                epochs=10,
                loss_fn=nn.MSELoss()
              ) -> None:
        
        if df is not None:
            epoch_array, time_array, lr_array, train_loss_array, validation_loss_array = self.get_arrays_from_df(df)
            start_epoch = epoch_array[-1]

        self.epoch_array = epoch_array
        self.time_array = time_array
        self.lr_array = lr_array
        self.train_loss_array = train_loss_array
        self.validation_loss_array = validation_loss_array

        for current_epoch in range(start_epoch + 1, start_epoch + 1 + epochs):
            print(f"\nepoch {current_epoch}\n-------------------------------")

            start_time = time.time()

            train_loss = train_loop(
                train_dataloader, model, loss_fn, optimizer)
            validation_loss = validation_loop(
                validation_dataloader, model, loss_fn)

            if scheduler is not None:
                print(
                    f"Learning rate (antes): {optimizer.param_groups[0]['lr']}")
                scheduler.step()
                print(
                    f"Learning rate (depois): {optimizer.param_groups[0]['lr']}")

            self.train_loss_array.append(train_loss)
            self.validation_loss_array.append(validation_loss)

            elapsed_time = time.time() - start_time

            self.epoch_array.append(current_epoch)
            self.time_array.append(elapsed_time)
            self.lr_array.append(optimizer.param_groups[0]['lr'])


def main():
    # Just some example code to test the functions

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 1e-3

    model_f2_5 = SRCNN(f2=5).to(device)
    model_f2_1 = SRCNN(f2=1).to(device)

    run_srcnn = ModelRunner()
    image_helper = ImageHelper()

    # torch.load('./results/srcnn/trained_models/model_f2_5.pth', map_location ='cpu')

    model_f2_5 = run_srcnn.load_model(
        model_f2_5, "./results/srcnn/trained_models/model_f2_5.pth")
    df_f2_5 = run_srcnn.load_df("./results/srcnn/dataframes/model_f2_5.csv")

    # model_f2_1 = run_srcnn.load_model(model_f2_1, "./results/srcnn/trained_models/model_f2_1.pth")
    # # df_f2_1 = run_srcnn.load_df("./results/srcnn/dataframes/model_f2_1.csv")

    run_srcnn.plot_train_validation_loss_from_df(df_f2_5)
    # # run_srcnn.plot_train_validation_loss_from_df(df_f2_1)

    # # run_srcnn.plot_train_validation_loss_comparision(
    # #     dfs=[
    # #         {"df": df_f2_5, "label": "f2_5"},
    # #         {"df": df_f2_1, "label": "f2_1"}
    # #     ]
    # # )


if __name__ == "__main__":
    main()

