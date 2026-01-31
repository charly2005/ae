import torch

from torchvision.datasets.utils import zipfile
from collections import OrderedDict
from torch.nn import Conv2d as Conv2d
from torch.nn import BatchNorm2d as BatchNorm2d
from torch.nn import ConvTranspose2d as ConvT2d
import torch.nn.functional as F
class AE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()

        assert hidden_dims[-1] == 2, "always use 2 as the latent dimension for generating a 2D image grid during evaluation"

        self.img_w = int(input_dim**0.5)
        self.hidden_dims = hidden_dims
        self.encoder = OrderedDict()
        self.decoder = OrderedDict()
        curr_ch = 1

        for i, h_dim in enumerate(hidden_dims):

          self.encoder[("conv",i)] = Conv2d(curr_ch, h_dim,kernel_size=3,stride=2,padding=1)
          curr_ch = h_dim

        reversed_dim = hidden_dims[::-1]
        for i, h_dim in enumerate(reversed_dim[1:] + [1]):

          self.decoder[("conv"),i] = ConvT2d(curr_ch, h_dim, kernel_size=3,stride=2,output_padding=1)
          curr_ch = h_dim

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        # from niantic labs
        self.encoder_list = torch.nn.ModuleList(list(self.encoder.values()))
        self.decoder_list = torch.nn.ModuleList(list(self.decoder.values()))

    def decode(self, x):
        for i in range(len(self.hidden_dims)):
          x = self.decoder[("conv",i)](x)
          if i < len(self.hidden_dims) -1:
            x = self.relu(x)
          else:
            x = self.sigmoid(x)
            if x.shape[-1] != self.img_w:
              # warp the image to specified size
              # align corners = false to avoid distortion
              # align corners = true is kind of like when you drag the corner of an image in photoshop, it distorts the image to fit that size
              x = F.interpolate(x, size=self.img_w, mode='bilinear', align_corners=False)
        return x

    def encode(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, self.img_w, self.img_w)
        for i in range(len(self.hidden_dims)):
          x = self.encoder[("conv", i)](x)
          # print(x.shape)

          if i < len(self.hidden_dims) - 1:
            x = self.relu(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        decoded = self.decode(z)
        return {"imgs": decoded}

