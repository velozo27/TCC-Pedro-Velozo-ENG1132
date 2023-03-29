import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter 

class Discriminator(nn.Module):
    def __init__(self, img_dim: int) -> None:
        super().__init__()

        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128), # take `in_features` into 128 nodes 
            nn.LeakyReLU(0.1), # Could also be ReLU here
            nn.Linear(128, 1), # output a single value if is real or fake (0 - fake, 1 - true)
            nn.Sigmoid() # Activation function to ensure the output value is between 0 and 1
        )

    def forward(self, x) -> nn.Sequential:
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim: int, img_dim: int) -> None:
        """"
        Args:
            z_dim (int): "Latents noise" or just noise
            img_dim (int): Image dimension
        """
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim), # img_dim == 784, because the sample data is 28x28x1 that once flattened is equal to 784
            nn.Tanh(), # Activation function to ensure the output of the pixel values are between -1 and 1. We do this because the input in this case will be normalized to also be in this range. It makes sense that the input and output be normalized in the same range
            )
    
    def forward(self, x) -> nn.Sequential:
        return self.gen(x)
    
# Hyperparameters etc.
# Note: GAN's are incredibly sensible to hyperparameters
# device = "cuda" if torch.cuda.is_available else "cpu"
device = "cpu"
lr = 3e-4 # learning rate
z_dim = 64
image_dim = 28 * 28 * 1 # 784
batch_size = 32
num_epochs = 50

disc = Discriminator(img_dim=image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)

# Creating noise to see how it changes through the epochs
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

# Mean STD of the MNist dataset = 0.1307, 0.3081
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
)

dataset = datasets.MNIST(root='study_datasets/GANs/simple_gan', transform=transforms, download=True)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# optmizer for the descriminator
opt_disc = optim.Adam(disc.parameters(), lr=lr)

# optmizer for the generator
opt_gen = optim.Adam(gen.parameters(), lr=lr)

# Loss function
criterion = nn.BCELoss()

# for tensor board
writer_fake = SummaryWriter(f"study_runs/GANs/GAN_MNIS/fake")
writer_real = SummaryWriter(f"study_runs/GANs/GAN_MNIS/real")
step = 0

# training
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device) # reshape data, -1 to keep the numbers of examples in our batch and flatten everything to 784
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(real)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        # generate fake image
        fake = gen(noise)

        # log(D(real)) part of the loss function
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        # log(1 - D(G(z))) part of the loss function
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator min log(1 - D(G(z))) <--> max log(D(G(z)))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        # Stuff for tensor board
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
