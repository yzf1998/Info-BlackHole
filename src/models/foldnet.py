import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from src.losses import EMD, SWD, Chamfer
from src.models.dgcnn import DGCNN



class Encoder(nn.Module):
    def __init__(self, num_points):
        super(Encoder, self).__init__()
        self.num_points = num_points
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024 + 64, 1024)
        self.fc2 = nn.Linear(1024, 512)

    def forward(self, input):
        input = input.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(input)))
        local_feature = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        global_feature = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        feature = torch.cat([local_feature, global_feature], 1)

        x = F.relu(self.fc1(feature.transpose(1, 2)))
        x = F.relu(self.fc2(x))

        return torch.max(x, 1, keepdim=True)[0]


class Decoder(nn.Module):
    def __init__(self, 
                 num_points=2048, 
                 m=2025):
        super(Decoder, self).__init__()
        self.n = num_points
        self.m = m
        self.grid_size = int(np.sqrt(self.m))
        self.meshgrid = [[-0.3, 0.3, self.grid_size], [-0.3, 0.3, self.grid_size]]
        self.mlp1 = nn.Sequential(
            nn.Conv1d(514, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1),
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(515, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1),
        )

    def build_grid(self, batch_size, device='cpu'):
        x = np.linspace(*self.meshgrid[0])
        y = np.linspace(*self.meshgrid[1])
        grid = np.array(list(itertools.product(x, y)))
        grid = np.repeat(grid[np.newaxis, ...], repeats=batch_size, axis=0)
        grid = torch.tensor(grid, device=device)
        return grid.float()

    def forward(self, input):
        device = input.device
        input = input.transpose(1, 2).repeat(1, 1, self.m)
        grid = self.build_grid(input.shape[0], device=device).transpose(1, 2)
        concate1 = torch.cat((input, grid), dim=1)
        after_folding1 = self.mlp1(concate1)
        concate2 = torch.cat((input, after_folding1), dim=1)
        after_folding2 = self.mlp2(concate2)
        return after_folding2.transpose(1, 2)


class FoldNet(nn.Module):
    def __init__(self, 
                 num_points=2048,
                 m=2025,
                 loss=None):
        super(FoldNet, self).__init__()

        self.num_projs = 1000
        self.num_points = num_points
        self.m = m
        assert self.m <= self.num_points
        self.grid_size = int(np.sqrt(self.m))
        assert (self.grid_size == 45 or self.grid_size == 32)
        self.latent_dim = 512

        self.encoder = DGCNN(emb_dims=self.latent_dim)
        self.decoder = Decoder(num_points=self.num_points,
                               m=self.m)
        self._loss_initialized = False
        self.swd_loss = None
        self.emd_loss = None
        self.cd_loss = None
        self.pmp_loss = None

    def encode(self, input):
        codeword, _ = self.encoder(input)
        return codeword

    def forward(self, input):
        codeword, source_ps_feat = self.encoder(input)
        output = self.decoder(codeword)

        return output

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def get_loss(self, input, output, regist_info=None):
        device = input.device
        if not self._loss_initialized:
            self.cd_loss  = Chamfer().to(device)      if hasattr(Chamfer(), 'to') else Chamfer()
            self.emd_loss = EMD()
            self.swd_loss = SWD(num_projs=1000, device=str(device))
            self.pmp_loss = nn.L1Loss().to(device)
            self._loss_initialized = True

        rep_num = self.num_points - self.m
        if rep_num:
            output_rep = output[:, -1, :].unsqueeze(1).repeat(1, rep_num, 1)
            output = torch.cat((output, output_rep), 1)
        if regist_info is not None:
            out_pred, delta_x = regist_info
            return self.cd_loss(input, output)["loss"], self.swd_loss(input, output)["loss"], self.cd_loss(out_pred, output)["loss"], self.pmp_loss(delta_x, torch.zeros_like(delta_x))
        return self.cd_loss(input, output)["loss"], self.swd_loss(input, output)["loss"]

class Stretch(nn.Module):
    from typing import Union
    def __init__(self, num_features: int, alpha_init: Union[float, torch.Tensor] = 0.2):
        super().__init__()
        shape = (1, num_features)
        if isinstance(alpha_init, torch.Tensor):
            assert list(alpha_init.shape) == [1, num_features], "`alpha_init` shape mismatch"
            alpha = alpha_init.clone().detach()
        else:
            alpha = torch.full(shape, float(alpha_init))
        self.register_buffer("alpha", alpha)

        self.gamma = nn.Parameter(0.01*torch.ones(shape))
        self.beta = nn.Parameter(np.pi*torch.ones(shape))
        self.register_buffer('moving_mag', 1.*torch.ones(shape))
        self.register_buffer('moving_min', np.pi*torch.ones(shape))

    def forward(self, X):
        if self.moving_mag.device != X.device:
            self.moving_mag = self.moving_mag.to(X.device)
            self.moving_min = self.moving_min.to(X.device)
            self.alpha = self.alpha.to(X.device)
        if self.training:
            min_ = X.min(dim=0, keepdim=True)[0]
            max_ = X.max(dim=0, keepdim=True)[0]
            mag_ = (max_ - min_).clamp_min(1e-8)
            self.moving_mag.mul_(0.99).add_(0.01 * mag_)
            self.moving_min.mul_(0.99).add_(0.01 * min_)
        else:
            min_, mag_ = self.moving_min, self.moving_mag

        X_hat = (X - min_) / mag_
        return (X_hat*self.gamma*self.alpha) + self.beta

class FoldNetDAE(nn.Module):
    def __init__(self, num_points=2048, m=2025, latent_dim=512):
        super(FoldNetDAE, self).__init__()
        self.num_projs = 1000
        self.num_points = num_points
        self.m = m
        assert self.m <= self.num_points
        self.grid_size = int(np.sqrt(self.m))
        assert (self.grid_size == 45 or self.grid_size == 32)
        self.latent_dim = latent_dim

        self.encoder = DGCNN(emb_dims=self.latent_dim)
        self.stretch = Stretch(self.latent_dim)
        self.to_dec = nn.Linear(self.latent_dim*2, self.latent_dim)
        self.decoder = Decoder(num_points=self.num_points,
                               m=self.m)
        
        self.swd_loss = SWD(self.num_projs, 'cuda:0')
        self.cd_loss = Chamfer()

    def reparameterize(self, z):
        if self.training:
            diff = torch.abs(z - z.unsqueeze(axis = 1))
            none_zeros = torch.where(diff == 0., torch.tensor([100.]).to(z.device), diff)
            z_scores,_ = torch.min(none_zeros, axis = 1)
            std =  torch.normal(mean = 0., std = 1.*z_scores).to(z.device)
            z = z + std
        c = torch.cat((torch.cos(2*np.pi*z), torch.sin(2*np.pi*z)), 1)
        return c
    
    def sample(self, num_samples = 100, z = None):
        c = torch.cat((torch.cos(2*np.pi*z), torch.sin(2*np.pi*z)), 1)
        codeword = F.leaky_relu(self.to_dec(c))
        samples = self.decoder(codeword)
        return samples
    
    def forward(self, input):
        codeword, source_ps_feat = self.encoder(input)
        s = self.stretch(codeword.squeeze(1))
        c = self.reparameterize(s)
        codeword = F.leaky_relu(self.to_dec(c))
        output = self.decoder(codeword.unsqueeze(1))
        return output

    def get_loss(self, input, output, regist_info=None):
        rep_num = self.num_points - self.m
        if rep_num:
            output_rep = output[:, -1, :].unsqueeze(1).repeat(1, rep_num, 1)
            output = torch.cat((output, output_rep), 1)
        if regist_info is not None:
            out_pred, delta_x = regist_info
            return self.cd_loss(input, output)["loss"], self.swd_loss(input, output)["loss"], self.cd_loss(out_pred, output)["loss"], self.pmp_loss(delta_x, torch.zeros_like(delta_x))
        return self.cd_loss(input, output)["loss"], self.swd_loss(input, output)["loss"]

if __name__ == "__main__":
    x = torch.randn(16, 1024, 3).cuda()
    model = FoldNetDAE(num_points=1024, m=1024)
    model.eval()
    model.cuda()
    y = model(x)
    chamfer = Chamfer()
    num_projs = 1000
    swd = SWD(num_projs=num_projs, device='cuda:0')
    emd = EMD()
    chamfer_value = chamfer.forward(x, y)["loss"].item()
    swd_value = np.sqrt(swd.forward(x, y)["loss"].item())
    swd_value = 0
    emd_value = emd.forward(x, y)["loss"].item()
    print('cd:', chamfer_value, ' swd:', swd_value, ' emd:', emd_value)



class Decoder22(nn.Module):
    def __init__(self, 
                 num_points=2048, 
                 m=2025,
                 latent_dim=512):
        super(Decoder, self).__init__()
        self.n = num_points
        self.m = m
        self.latent_dim = latent_dim
        self.grid_size = int(np.sqrt(self.m))
        self.meshgrid = [[-0.3, 0.3, self.grid_size], [-0.3, 0.3, self.grid_size]]

        in_dim1 = self.latent_dim + 2
        in_dim2 = self.latent_dim + 3

        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_dim1, 256, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 3, 1)
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(in_dim2, 256, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 3, 1)
        )

    def build_grid(self, batch_size):
        x = np.linspace(*self.meshgrid[0])
        y = np.linspace(*self.meshgrid[1])
        grid = np.array(list(itertools.product(x, y)))
        grid = np.repeat(grid[np.newaxis, ...], repeats=batch_size, axis=0)
        grid = torch.tensor(grid)
        return grid.float()

    def forward(self, codeword):
        codeword = codeword.transpose(1, 2).repeat(1, 1, self.m)
        grid = self.build_grid(codeword.shape[0]).transpose(1, 2)
        if torch.cuda.is_available():
            grid = grid.cuda()
        concate1 = torch.cat((codeword, grid), dim=1)
        after_folding1 = self.mlp1(concate1)
        concate2 = torch.cat((codeword, after_folding1), dim=1)
        after_folding2 = self.mlp2(concate2)
        return after_folding2.transpose(1, 2)
