import torch
import torch.nn as nn

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
        self.hidden_size = configs.hidden_size
        self.num_layers = configs.num_layers

        self.d_model = configs.d_embed
        self.n_heads = configs.n_heads

        self.seq_len = configs.seq_len
        self.pre_len = configs.pre_len

        # 横向注意力
        self.feature_size = configs.input_size

        self.gru = nn.GRU(self.seq_len, self.seq_len, self.num_layers, batch_first=True)
        # self.attn_layer = AttentionLayer(FullAttention(attention_dropout=0.1, output_attention=True),
        #                                  self.hidden_size, self.n_heads)
        self.attn_layer = AttentionLayer(FullAttention(attention_dropout=0, output_attention=True),
                                         self.feature_size, self.n_heads, d_keys=64, d_values=64, )
        self.fc = nn.Linear(self.hidden_size, self.pre_len)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.seq_len).to(x.device)
        # 前向传播 GRU
        x = x.transpose(1, 2)
        out, hidden = self.gru(x, h0)
        out = out + x  # 加残差效果好
        out = out.transpose(1, 2)  # out [Batch_size, Channel, Pred_len]
        # x shape: [Batch, Pred_len, Channel]
        # 前向传播 GRU
        context_vector, attention_g = self.attn_layer(out, out, out)

        # 全连接层输出
        context_vector = context_vector.transpose(1, 2)
        output = self.fc(context_vector)
        output = output.transpose(1, 2)
        return output
