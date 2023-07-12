import torch
import torch.nn as nn
import torch.optim as optim

# Define the VQGAN model
class VQGAN(nn.Module):
    def __init__(self, in_channels, codebook_size, codebook_dim):
        super(VQGAN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, codebook_dim, kernel_size=1, stride=1, padding=0)
        )

        self.codebook_size = codebook_size
        self.embedding = nn.Embedding(codebook_size, codebook_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(codebook_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, in_channels, kernel_size=4, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        latent = self.encoder(x)
        batch_size, _, H, W, D = latent.size()
        latent = latent.view(batch_size, -1, self.codebook_dim)

        # Quantize the latent codes
        _, quantized_idx = torch.min(torch.norm(latent[:, :, None, :] - self.embedding.weight[None, None, :, :], dim=-1), dim=-1)
        quantized = self.embedding(quantized_idx)

        reconstructed = self.decoder(quantized.view(batch_size, H, W, D, self.codebook_dim).permute(0, 4, 1, 2, 3))
        return reconstructed, quantized_idx

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = self.conv5(x)
        x = self.sigmoid(x)
        return x

# Define the training loop
def train_vqgan(model, discriminator, dataloader, num_epochs, learning_rate):
    criterion_recon = nn.MSELoss()
    criterion_adv = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    discriminator.to(device)

    for epoch in range(num_epochs):
        running_recon_loss = 0.0
        running_adv_loss = 0.0

        for i, data in enumerate(dataloader):
            inputs = data.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs, _ = model(inputs)

            # Reconstruction Loss
            recon_loss = criterion_recon(outputs, inputs)

            # Adversarial Loss
            real_labels = torch.ones(inputs.size(0), 1).to(device)
            fake_labels = torch.zeros(inputs.size(0), 1).to(device)

            real_preds = discriminator(inputs)
            fake_preds = discriminator(outputs.detach())

            adv_loss = criterion_adv(real_preds, real_labels) + criterion_adv(fake_preds, fake_labels)

            # Total Loss
            total_loss = recon_loss + adv_loss

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            # Accumulate the losses
            running_recon_loss += recon_loss.item()
            running_adv_loss += adv_loss.item()

            # Print the average losses every few iterations
            if i % 10 == 9:
                print(f"[Epoch {epoch+1}, Iteration {i+1}] Recon Loss: {running_recon_loss / 10}, Adv Loss: {running_adv_loss / 10}")
                running_recon_loss = 0.0
                running_adv_loss = 0.0

# Usage example
# Assuming you have a voxel grid dataset and dataloader
# Initialize the model and discriminator
model = VQGAN(in_channels=1, codebook_size=256, codebook_dim=64)
discriminator = Discriminator(in_channels=1)

# Define the hyperparameters
num_epochs = 10
learning_rate = 0.001

# Train the model
train_vqgan(model, discriminator, dataloader, num_epochs, learning_rate)
