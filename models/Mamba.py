import torch.nn as nn

import torch.nn as nn
import torch
from einops import rearrange, repeat, einsum
from torch.nn.utils import weight_norm


class SSM(nn.Module):
    def __init__(self, N, D, L):
        super(SSM, self).__init__()
        self.L = L
        self.D = D
        self.N = N
        # 创建A矩阵
        A = repeat(torch.arange(1, N + 1), 'n -> d n', d=D)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(D))
        self.s_b = nn.Linear(D, N)
        self.s_c = nn.Linear(D, N)
        self.s_1 = nn.Linear(D, 1)
        self.s_delta = nn.Linear(1, D)
        self.softplus = nn.Softplus()

    def forward(self, x):
        """

        :param x: (B,L,D)
        :return: y(B,L,D)
        """

        (b, l, d) = x.shape
        n = self.N

        A = -torch.exp(self.A_log.float())  # shape (D,N)
        D = self.D.float()
        B = self.s_b(x)
        C = self.s_c(x)
        delta = self.softplus(self.s_delta(self.s_1(x)))

        # 离散化A和B 见论文公式（4）
        deltaA = torch.exp(einsum(delta, A, 'b l d, d n -> b l d n'))
        deltaB_u = einsum(delta, B, x, 'b l d, b l n, b l d  -> b l d n')

        h = torch.zeros((b, d, n), device=deltaA.device)
        ys = []
        for i in range(l):
            h = deltaA[:, i] * h + deltaB_u[:, i]
            y = einsum(h, C[:, i, :], 'b d n, b n -> b d')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d)

        y = y

        return y


class Mamba_block(nn.Module):
    def __init__(self, L, n_layers, D, N, feature_size):
        super(Mamba_block, self).__init__()
        self.L = L
        self.n_layers = n_layers
        self.D = D
        self.N = N
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.linear_x1 = nn.Linear(feature_size, D)
        self.linear_x2 = nn.Linear(feature_size, D)
        # [B, L, D]
        self.conv = nn.Conv1d(D, D, 1)
        self.ssm = SSM(N, D, L)
        self.linear_x3 = nn.Linear(D, feature_size)

    def forward(self, input):
        # x =input
        # res_x = input
        x = self.linear_x1(input)
        res_x = self.silu(self.linear_x2(input))

        x = x.transpose(2, 1)
        x = self.conv(x)
        x = x.transpose(2, 1)
        x = self.silu(x)
        x = self.ssm(x)

        # 使用 torch.mul() 函数进行同位置相乘
        out = torch.mul(x, res_x)

        y = self.linear_x3(out)
        return y


class Model(nn.Module):  # Mamba
    # L, n_layers, D, N
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pre_len = configs.pre_len
        self.num_layers = configs.num_layers
        self.feature_size = configs.input_size
        D = 64
        N = 128
        layers = []

        for i in range(self.num_layers):
            layers += [Mamba_block(self.seq_len, self.num_layers, D, N, self.feature_size)]

        self.network = nn.Sequential(*layers)
        self.out = nn.Linear(self.seq_len, self.pre_len)

    def forward(self, x):
        x = self.network(x)
        x = x.transpose(1, 2)
        x = self.out(x).transpose(1, 2)
        return x
