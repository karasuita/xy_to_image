import torch
import torch.nn as nn

from module import Module


class Model(Module):

    def __init__(self, nz, nz2=64):
        super().__init__()

        self.nz = nz
        self.nz2 = nz2

        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz * 2, nz2 * 8, 4, 1, 0, bias=True),
            nn.BatchNorm2d(nz2 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(nz2 * 8, nz2 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nz2 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(nz2 * 4, nz2 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nz2 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(nz2 * 2, nz2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nz2),
            nn.ReLU(True),
            nn.ConvTranspose2d(nz2, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
