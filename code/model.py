import torch
import torch.nn as nn
import torch.nn.functional as F

# Flow: input img -> hidden dim -> mean, std -> reparametrization trick -> output img
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=400, z_dim=20):
        super(VariationalAutoEncoder, self).__init__()
        # Encoder
        self.img2hid = nn.Linear(input_dim, hidden_dim)
        self.hid1 = nn.Linear(hidden_dim, hidden_dim)
        self.hid2mean = nn.Linear(hidden_dim, z_dim)
        self.hid2log_var = nn.Linear(hidden_dim, z_dim)
        # Decoder
        self.z2hid = nn.Linear(z_dim, hidden_dim)
        self.hid2 = nn.Linear(hidden_dim, hidden_dim)
        self.hid2img = nn.Linear(hidden_dim, input_dim)

    def encode(self,x):
        # q_phi(z|x)
        h = F.relu(self.img2hid(x))
        h = F.relu(self.hid1(h))
        mu, log_var = self.hid2mean(h), self.hid2log_var(h)
        return mu, log_var

    def decode(self,z):
        # p_theta(x|z)
        h = F.relu(self.z2hid(z))
        h = F.relu(self.hid2(h))
        return torch.sigmoid(self.hid2img(h))   # ensure o/p btw [0,1] since MNIST images were normalzed

    def forward(self,x):
        mu, log_var = self.encode(x)
        var = torch.exp(0.5 * log_var)
        z_reparam = mu + var * torch.randn_like(var)
        x_reconstructed = self.decode(z_reparam)
        return x_reconstructed, mu, log_var

class VAELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(VAELoss, self).__init__()
        self.reduction = reduction

    def forward(self, reconstructed_x, x, mu, log_var):
        reconstruction_loss = F.binary_cross_entropy(reconstructed_x, x, reduction=self.reduction)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return reconstruction_loss + kl_divergence, reconstruction_loss, kl_divergence