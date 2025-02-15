import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class Conditioner(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)

class CTGANGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.conditioner = Conditioner(output_dim, hidden_dim)
        
    def forward(self, z, c):
        x = self.net(z)
        h = self.conditioner(c)
        return x * h

class CTGANDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

class CTGAN:
    def __init__(self, input_dim, latent_dim=100):
        self.latent_dim = latent_dim
        self.generator = CTGANGenerator(latent_dim, input_dim)
        self.discriminator = CTGANDiscriminator(input_dim)
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=2e-4)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4)
        
    def train_step(self, real_data, modes):
        batch_size = real_data.shape[0]
        
        # Train Discriminator
        z = torch.randn(batch_size, self.latent_dim)
        fake_data = self.generator(z, real_data)
        
        real_pred = self.discriminator(real_data)
        fake_pred = self.discriminator(fake_data.detach())
        
        d_loss = -torch.mean(torch.log(real_pred + 1e-8) + torch.log(1 - fake_pred + 1e-8))
        
        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        z = torch.randn(batch_size, self.latent_dim)
        fake_data = self.generator(z, real_data)
        fake_pred = self.discriminator(fake_data)
        
        g_loss = -torch.mean(torch.log(fake_pred + 1e-8))
        
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()
        
        return d_loss.item(), g_loss.item()

    def train(self, data, epochs=200, batch_size=500):
        for epoch in range(epochs):
            d_losses = []
            g_losses = []
            
            idx = np.random.permutation(len(data))
            data = data[idx]
            
            for i in range(0, len(data), batch_size):
                batch = torch.FloatTensor(data[i:i+batch_size])
                d_loss, g_loss = self.train_step(batch, None)
                d_losses.append(d_loss)
                g_losses.append(g_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: D_loss={np.mean(d_losses):.4f}, G_loss={np.mean(g_losses):.4f}')
    
    def generate(self, num_samples, condition_data):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            samples = self.generator(z, condition_data)
        return samples.numpy()
    


# Good, but with a number of hyperparameters to tune, such as the latent dimension, the learning_rate, 
# the dimension of the generator, discriminator, etc...
# As suggested by ChatGPT, we could try a strategy of GridSearch and/or Random Search to optimize the 
# hyperparameters, but we would need the data to do it.