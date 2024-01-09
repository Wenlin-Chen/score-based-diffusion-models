import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):

    def __init__(self, t_embed_dim, scale=30.0):
        super().__init__()

        self.register_buffer("w", torch.randn(t_embed_dim//2)*scale)

    def forward(self, t):
        # t: (B, )
        t_proj = 2.0 * torch.pi * self.w[None, :] * t[:, None]  # (B, E//2)
        t_embed = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)  # (B, E)
        return t_embed
    

class TimeProjectionDense(nn.Module):

    def __init__(self, t_embed_dim, n_channels):
        super().__init__()

        self.net = nn.Linear(t_embed_dim, n_channels)

    def forward(self, t_embed):
        # t_embed: (B, E)
        return self.net(t_embed)[..., None, None]  # (B, C, 1, 1)
    

class TimeConditionalScoreNet(nn.Module):

    def __init__(
            self, 
            in_channels,
            forward_pdf_std,
            sigma,
            conv_channels=[32, 64, 128, 256], 
            kernel_sizes=[3, 3, 3, 3],
            strides=[1, 2, 2, 2],
            paddings=[0, 0, 0, 0],
            output_paddings=[0, 1, 1, 0],
            t_embed_dim=256,
            groups=[4, 32, 32, 32],
            conv_bias=False,
        ):
        super().__init__()

        self.t_embedding = nn.Sequential(
            TimeEmbedding(t_embed_dim),
            nn.Linear(t_embed_dim, t_embed_dim),
        )

        # Encoder
        self.conv1 = nn.Conv2d(in_channels, conv_channels[0], kernel_sizes[0], strides[0], paddings[0], bias=conv_bias)
        self.tproj1 = TimeProjectionDense(t_embed_dim, conv_channels[0])
        self.groupnorm1 = nn.GroupNorm(groups[0], conv_channels[0])
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_sizes[1], strides[1], paddings[1], bias=conv_bias)
        self.tproj2 = TimeProjectionDense(t_embed_dim, conv_channels[1])
        self.groupnorm2 = nn.GroupNorm(groups[1], conv_channels[1])
        self.conv3 = nn.Conv2d(conv_channels[1], conv_channels[2], kernel_sizes[2], strides[2], paddings[2], bias=conv_bias)
        self.tproj3 = TimeProjectionDense(t_embed_dim, conv_channels[2])
        self.groupnorm3 = nn.GroupNorm(groups[2], conv_channels[2])
        self.conv4 = nn.Conv2d(conv_channels[2], conv_channels[3], kernel_sizes[3], strides[3], paddings[3], bias=conv_bias)
        self.tproj4 = TimeProjectionDense(t_embed_dim, conv_channels[3])
        self.groupnorm4 = nn.GroupNorm(groups[3], conv_channels[3])

        # Decoder
        self.d_conv1 = nn.ConvTranspose2d(conv_channels[-1], conv_channels[-2], kernel_sizes[-1], strides[-1], paddings[-1], output_paddings[0], bias=conv_bias)
        self.d_tproj1 =TimeProjectionDense(t_embed_dim, conv_channels[-2])
        self.d_groupnorm1 = nn.GroupNorm(groups[-1], conv_channels[-2])
        self.d_conv2 = nn.ConvTranspose2d(conv_channels[-2]*2, conv_channels[-3], kernel_sizes[-2], strides[-2], paddings[-2], output_paddings[1], bias=conv_bias)
        self.d_tproj2 =TimeProjectionDense(t_embed_dim, conv_channels[-3])
        self.d_groupnorm2 = nn.GroupNorm(groups[-2], conv_channels[-3])
        self.d_conv3 = nn.ConvTranspose2d(conv_channels[-3]*2, conv_channels[-4], kernel_sizes[-3], strides[-3], paddings[-3], output_paddings[2], bias=conv_bias)
        self.d_tproj3 =TimeProjectionDense(t_embed_dim, conv_channels[-4])
        self.d_groupnorm3 = nn.GroupNorm(groups[-3], conv_channels[-4])
        self.d_conv4 = nn.ConvTranspose2d(conv_channels[-4]*2, in_channels, kernel_sizes[-4], strides[-4], paddings[0], output_paddings[3], strides[-4])

        self.act = lambda x: x * torch.sigmoid(x)
        self.forward_pdf_std = forward_pdf_std
        self.register_buffer("sigma", sigma)

    def forward(self, x, t):
        # x: (B, C_in, H, W)
        # t: (B, )
        # std: (B, 1, 1, 1)

        t_embed = self.act(self.t_embedding(t))

        h1 = self.conv1(x) + self.tproj1(t_embed)
        h1 = self.groupnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1) + self.tproj2(t_embed)
        h2 = self.groupnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2) + self.tproj3(t_embed)
        h3 = self.groupnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3) + self.tproj4(t_embed)
        h4 = self.groupnorm4(h4)
        h4 = self.act(h4)

        h = self.d_conv1(h4) + self.d_tproj1(t_embed)
        h = self.d_groupnorm1(h)
        h = self.act(h)
        h = self.d_conv2(torch.concat([h, h3], dim=1)) + self.d_tproj2(t_embed)
        h = self.d_groupnorm2(h)
        h = self.act(h)
        h = self.d_conv3(torch.concat([h, h2], dim=1)) + self.d_tproj3(t_embed)
        h = self.d_groupnorm3(h)
        h = self.act(h)
        h = self.d_conv4(torch.concat([h, h1], dim=1))
        
        std = self.forward_pdf_std(t, self.sigma)[:, None, None, None]
        h = h / std
        return h
    
    def compute_loss(self, x, eps=1e-5):
        # x: (B, C_in, H, W)
        # sigma: (,)

        t = torch.rand(x.shape[0], device=x.device) * (1.0-eps) + eps  # (B, )
        z = torch.randn_like(x)
        std = self.forward_pdf_std(t, self.sigma)[:, None, None, None]  # (B, 1, 1, 1)
        x_t = x + z * std  # (B, C_in, H, W)
        score = self(x_t, t) 
        loss = torch.mean(torch.sum((score*std+z)**2, dim=(1, 2, 3)))

        return loss
