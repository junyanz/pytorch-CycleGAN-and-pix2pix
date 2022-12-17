import torch
import torch.nn as nn
import torch.nn.functional as func


class BasisConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            basis_dim: int = 7,
            hidden_dim: int = 10
    ):
        super().__init__()

        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._basis_dim = basis_dim
        self._hidden_dim = hidden_dim

        self._basis_generator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, kernel_size * kernel_size),
        )
        weights_a = torch.empty(out_channels, in_channels, basis_dim)
        self._a = nn.Parameter(nn.init.uniform_(weights_a), requires_grad=True)

    def forward(self, x):
        weights = self._get_weights()
        return func.conv2d(x,
                           weights, stride=self._stride,
                           dilation=self._dilation,
                           padding=self._dilation)

    def _get_weights(self):
        z = torch.randn(self._basis_dim, self._hidden_dim)
        basis = self._basis_generator(z)
        weights = self._a @ basis
        return weights.view(self._out_channels, self._in_channels,
                            self._kernel_size, self._kernel_size)


class BasisTransposeConv2d(BasisConv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            basis_dim: int = 7,
            hidden_dim: int = 10
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            basis_dim=basis_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, x):
        weights = self._get_weights()
        return func.conv_transpose2d(x,
                                     weights, stride=self._stride,
                                     dilation=self._dilation,
                                     padding=self._dilation)
