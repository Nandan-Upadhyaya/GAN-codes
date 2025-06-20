import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.generator.fusion_block import AffineBlock


class ResidualBlockG(nn.Module):
    def __init__(self, df_1_c_out: int, df_2_c_out: int, affine_h_d: int = 256,
                 df_1_c_in: int = 256, df_2_c_in: int = 256):
        super().__init__()
        # DFBlock_1
        self.fusion_block_1 = AffineBlock(input_dim=df_1_c_in, hidden_dim=affine_h_d, output_dim=df_1_c_out)
        self.fusion_block_2 = AffineBlock(input_dim=df_1_c_in, hidden_dim=affine_h_d, output_dim=df_1_c_out)
        self.conv_1 = nn.Conv2d(df_1_c_out, df_2_c_out, kernel_size=3, stride=1, padding=1)

        # DFBlock_2
        self.fusion_block_3 = AffineBlock(input_dim=df_2_c_in, hidden_dim=affine_h_d, output_dim=df_2_c_out)
        self.fusion_block_4 = AffineBlock(input_dim=df_2_c_in, hidden_dim=affine_h_d, output_dim=df_2_c_out)
        self.conv_2 = nn.Conv2d(df_2_c_out, df_2_c_out, kernel_size=3, stride=1, padding=1)

        self.scale_conv = None
        if df_1_c_out != df_2_c_out:
            self.scale_conv = nn.Conv2d(df_1_c_out, df_2_c_out, kernel_size=1, stride=1, padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))

    def _shortcut(self, x: Tensor) -> Tensor:
        if self.scale_conv is not None:
            x = self.scale_conv(x)

        return x

    def _df_block_1(self, x: Tensor, sentence_embed: Tensor) -> Tensor:
        h = self.fusion_block_1(x, sentence_embed)
        h = F.leaky_relu(h, 0.2, inplace=True)
        h = self.fusion_block_2(h, sentence_embed)
        h = F.leaky_relu(h, 0.2, inplace=True)
        return self.conv_1(h)

    def _df_block_2(self, x: Tensor, sentence_embed: Tensor) -> Tensor:
        h = self.fusion_block_3(x, sentence_embed)
        h = F.leaky_relu(h, 0.2, inplace=True)
        h = self.fusion_block_4(h, sentence_embed)
        h = F.leaky_relu(h, 0.2, inplace=True)
        return self.conv_2(h)

    def _residual(self, x: Tensor, y):
        # DFBlock_1
        h_1 = self._df_block_1(x, y)

        # DFBlock_2
        h_2 = self._df_block_2(h_1, y)

        return h_2

    def forward(self, x, sentence_embed: Tensor) -> Tensor:
        return self._shortcut(x) + self.gamma * self._residual(x, sentence_embed)