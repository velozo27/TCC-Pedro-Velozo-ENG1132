import time
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
from torchmetrics import PeakSignalNoiseRatio
from torchvision import transforms
import tqdm
from PIL import Image
from loops import train_loop, validation_loop


class SRCNN(nn.Module):
    def __init__(self, f2=5):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=(2, 2))
        self.conv2 = nn.Conv2d(64, 32, kernel_size=f2, padding=(2, 2))
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=(2, 2))

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

    def get_metrics(self) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
        return self.epoch_array, self.time_array, self.lr_array, self.train_loss_array, self.validation_loss_array
                    
    def load_model(self, model: nn.Module, model_weights_path: str) -> None:
        model.load_state_dict(torch.load(model_weights_path))

    def save_model_weights(self, model: nn.Module, model_weights_path: str) -> None:
        torch.save(model.state_dict(), model_weights_path)

    def calculate_psnr(self,
                       img1: torch.Tensor,
                       img2: torch.Tensor) -> float:
        psnr = PeakSignalNoiseRatio()
        return psnr(img1, img2)

    def compare_models(self,
                       models: list[dict[str, nn.Module]],
                       images_path: str
                       ) -> pd.DataFrame:

        pathlist = Path(images_path).rglob('*.png')
        number_of_images = len(list(pathlist))

        psnr_array = []
        bicubic_psnr_array = []

        transform = transforms.Compose([
            transforms.ToTensor(),
            # resize image to 33x33 and downsample by BICUBIC interpolation
            transforms.Resize(
                (image.size[1] // 3, image.size[0] // 3), interpolation=Image.BICUBIC),
            # resize image to 256x256
            transforms.Resize(
                (image.size[1], image.size[0]), interpolation=Image.BICUBIC)
        ])

        for model_dict in models:
            model_psnr_avg = 0
            bicubic_psnr_avg = 0

            model_name = model_dict["name"]
            model = model_dict["model"]

            for img_path in tqdm(images_path):
                path_in_str = str(img_path)
                image = Image.open(path_in_str)

                bicubic_image = transform(image)
                model_image = model(bicubic_image.to(self.device))

                # target for PSNR metric
                targets = transforms.ToTensor()(image)

                # bicubic PSNR
                preds = bicubic_image
                bicubic_psnr_avg += self.calculate_psnr(preds, targets)

                # model PSNR
                preds = model_image
                model_psnr_avg += self.calculate_psnr(preds, targets)

            bicubic_psnr_avg /= number_of_images
            model_psnr_avg /= number_of_images

            psnr_array.append({
                model_name:
                [model_psnr_avg.detach().numpy()]})

            bicubic_psnr_array.append({
                model_name: [bicubic_psnr_avg.detach().numpy()]})

        psnr_df = pd.DataFrame(psnr_array)
        psnr_df.set_index('', inplace=True)
        bicubic_psnr_df = pd.DataFrame(bicubic_psnr_array)
        bicubic_psnr_df.set_index('', inplace=True)

        return pd.concat([psnr_df, bicubic_psnr_df], axis=1)

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
