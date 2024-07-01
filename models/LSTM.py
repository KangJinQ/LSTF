import torch
import torch.nn as nn


# 定义LSTM模型
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.hidden_size = configs.hidden_size
        self.input_size = configs.input_size
        self.num_layers = configs.num_layers
        self.outputs = configs.output_size

        self.seq_len = configs.seq_len
        self.pre_len = configs.pre_len

        self.lstm = nn.LSTM(self.seq_len, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.pre_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)  # lstm outputs: output,(h_n,c_n)  shape:(L,N,D∗H_out),((D*num_layers,N,H_out),(D*num_layers,N,H_cell))
        out = self.fc(out)
        # 取最后一个时间步的输出，作为线性层的输入
        out = out.transpose(1, 2)
        output = out[:, -self.pre_len:, :]
        # out = self.fc(out)
        return output
