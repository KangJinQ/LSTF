import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from models.attn import FullAttention, AttentionLayer


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.num_layers = configs.num_layers

        self.d_embed = configs.d_embed
        self.n_heads = configs.n_heads

        self.seq_len = configs.seq_len
        self.pre_len = configs.pre_len

        # 横向注意力
        self.feature_size = configs.input_size
        self.factor_x = configs.factor_x
        # self.delayer_dim = configs.delayer_dim

        self.decompsition = series_decomp(self.factor_x + 1)  # 5
        # self.decompsition2 = series_decomp(25)  # 25
        self.decompsition3 = series_decomp(self.seq_len//self.factor_x + 1)  # 65

        # self.layers = []
        # for i in range(len(self.delayer_dim)):
        #     # factor_i = 2 ** (i+1)
        #     self.layers += [series_decomp(self.delayer_dim[i])]

        self.gru = nn.GRU(self.seq_len, self.seq_len, self.num_layers, batch_first=True)
        # self.ft_attn_layer = AttentionLayer(FullAttention(attention_dropout=0.6, output_attention=True),
        #                                     self.seq_len, self.n_heads, d_keys=256, d_values=256, )
        self.attn_layer = AttentionLayer(FullAttention(attention_dropout=0.1, output_attention=True),
                                         # self.feature_size+(len(self.delayer_dim))*2, self.n_heads,
                                         self.feature_size + 4, self.n_heads,
                                         d_keys=self.d_embed // self.n_heads, d_values=self.d_embed // self.n_heads, )
        # self.norm_layer = nn.LayerNorm(self.feature_size + 6)
        # self.relu = nn.ReLU()
        self.fc = nn.Linear(self.seq_len, self.pre_len)

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x[:, :, -1].unsqueeze(2))
        # seasonal_init2, trend_init2 = self.decompsition2(x[:, :, -1].unsqueeze(2))  , seasonal_init2, trend_init2
        seasonal_init3, trend_init3 = self.decompsition3(x[:, :, -1].unsqueeze(2))
        x = torch.cat((x, seasonal_init, trend_init, seasonal_init3, trend_init3), dim=2)

        x = x.transpose(1, 2)

        # # 测试这一部分——————————————————————————
        # o, attention_ft = self.ft_attn_layer(x, x, x)  # 横向特征注意力
        # #
        # o = o + x  # 残差
        # # 测试这一部分——————————————————————————

        h0 = torch.zeros(self.num_layers, x.size(0), self.seq_len).to(x.device)
        out, hidden = self.gru(x, h0)
        out = out + x  # 加残差效果好  out [Batch_size, Channel, GRU的第二个参数(目前是seq_len)]

        out = out.transpose(1, 2)  # out [Batch_size, Pred_len, Channel]
        # x shape: [Batch, Pred_len, Channel]
        out, attention_g = self.attn_layer(out, out, out)

        out = self.fc(out.transpose(1, 2))
        output = out.transpose(1, 2)
        return output[:, :, :self.feature_size], attention_g
