import time
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
from torchmetrics import PeakSignalNoiseRatio
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from loops import train_loop, validation_loop
from torchmetrics.image import StructuralSimilarityIndexMeasure


class SRCNN(nn.Module):
    def __init__(self, f2=1):
        super(SRCNN, self).__init__()
        padding = [2,2,2]
        if f2 == 5:
            padding = [9 // 2, 5 // 2, 5 // 2]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=padding[0])
        self.conv2 = nn.Conv2d(64, 32, kernel_size=f2, padding=padding[1])
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=padding[2])

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class RunSRCNN():
    def __init__(self, model=None, device="cuda" if torch.cuda.is_available() else "cpu", train_loop=train_loop, validation_loop=validation_loop):
        self.device = device

        self.model_f2_5 = SRCNN(f2=5).to(device)
        self.model_f2_1 = SRCNN(f2=1).to(device)

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

    def get_metrics(self) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
        return self.epoch_array, self.time_array, self.lr_array, self.train_loss_array, self.validation_loss_array
                    
    def load_model(self, model: nn.Module, model_weights_path: str) -> None:
        model.load_state_dict(torch.load(model_weights_path))

    def save_model_weights(self, model: nn.Module, model_weights_path: str) -> None:
        torch.save(model.state_dict(), model_weights_path)

    def calculate_psnr(self,
                       img1: torch.Tensor,
                       img2: torch.Tensor) -> float:
        psnr = PeakSignalNoiseRatio().to(self.device)
        return psnr(img1.to(self.device)
        , img2.to(self.device))

    def calculate_ssim(self,
                        img1: torch.Tensor,
                        img2: torch.Tensor) -> float:

        # add batch dimension, since ssim expects a batch of images and not a single image. And unsqueeze adds a dimension at the specified position
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)                
                    
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        return ssim(img1.to(self.device), img2.to(self.device))

    def compare_models(self,
                       models: list[dict[str, nn.Module]],
                       images_path: str
                       ) -> pd.DataFrame:

        pathlist = Path(images_path).rglob('*.png')
        number_of_images = len(list(pathlist))
    
        psnr_dict = {}
        bicubic_psnr_dict = {}
        ssim_dict = {}
        bicubic_ssim_dict = {}

        for model_dict in models:
            model_psnr_avg = 0
            bicubic_psnr_avg = 0
            model_ssim_avg = 0
            bicubic_ssim_avg = 0

            model_name = model_dict["name"]
            model = model_dict["model"]

            pathlist = Path(images_path).rglob('*.png')
    
            for img_path in tqdm(pathlist):
                path_in_str = str(img_path)
                
                image = Image.open(path_in_str)
                print(image)

                transform = transforms.Compose([
                    transforms.ToTensor(),
                    # resize image to 33x33 and downsample by BICUBIC interpolation
                    transforms.Resize(
                        (image.size[1] // 3, image.size[0] // 3), interpolation=Image.BICUBIC),
                    # resize image to 256x256
                    transforms.Resize(
                        (image.size[1], image.size[0]), interpolation=Image.BICUBIC)
                ])

                bicubic_image = transform(image)
                model_image = model(bicubic_image.to(self.device))

                # target for PSNR metric
                targets = transforms.ToTensor()(image).to(self.device)

                # bicubic PSNR
                preds = bicubic_image.to(self.device)
                bicubic_psnr_avg += self.calculate_psnr(preds, targets)
                bicubic_ssim_avg += self.calculate_ssim(preds, targets)

                # model PSNR
                preds = model_image.to(self.device)
                model_psnr_avg += self.calculate_psnr(preds, targets)
                model_ssim_avg += self.calculate_ssim(preds, targets)

            bicubic_psnr_avg /= number_of_images
            model_psnr_avg /= number_of_images
            bicubic_ssim_avg /= number_of_images
            model_ssim_avg /= number_of_images

            psnr_dict[model_name] = model_psnr_avg.item() 
            bicubic_psnr_dict[model_name] = bicubic_psnr_avg.item()
            ssim_dict[model_name] = model_ssim_avg.item()
            bicubic_ssim_dict[model_name] = bicubic_ssim_avg.item()

        psnr_df = pd.DataFrame([psnr_dict], index=['PSNR'])        
        bicubic_psnr_df = pd.DataFrame([bicubic_psnr_dict], index=['Bicubic PSNR'])
        ssim_df = pd.DataFrame([ssim_dict], index=['SSIM'])
        bicubic_ssim_df = pd.DataFrame([bicubic_ssim_dict], index=['Bicubic SSIM'])
    
        return pd.concat([psnr_df, bicubic_psnr_df, ssim_df, bicubic_ssim_df])

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
              epochs=10, loss_fn=nn.MSELoss()
              ) -> None:

        for current_epoch in range(epochs):
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 1e-3

    model_f2_5 = SRCNN(f2=5).to(device)
    optimizer_f2_5 = torch.optim.Adam(model_f2_5.parameters(), lr=lr)
    scheduler_f2_5 = torch.optim.lr_scheduler.LinearLR(
    optimizer_f2_5,
    start_factor=1.0,
    end_factor=0.01,
    total_iters=10)

    model_f2_1 = SRCNN(f2=1).to(device)
    optimizer_f2_1 = torch.optim.Adam(model_f2_1.parameters(), lr=lr)
    scheduler_f2_1 = torch.optim.lr_scheduler.LinearLR(
    optimizer_f2_1,
    start_factor=1.0,
    end_factor=0.01,
    total_iters=10)

    run_srcnn = RunSRCNN()
    path = f"./datasets/Set14/"
    df = run_srcnn.compare_models(
        [{'name': 'model_f2_1', 'model': model_f2_1}, {'name': 'model_f2_5', 'model': model_f2_5}],
        path,
    )

    print(df)

    


if __name__ == "__main__":
    main()
 