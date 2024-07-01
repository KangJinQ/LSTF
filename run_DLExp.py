import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

my_seed = 615
random.seed(my_seed)
torch.manual_seed(my_seed)
np.random.seed(my_seed)

parser = argparse.ArgumentParser(description='Deep Learning Methods for Time Series Forecasting')

# basic config
parser.add_argument('-model', type=str, default='Mamba', help="[TCN, 2DATCN, Linear, DLinear, LSTM, GRU, DAGRU"
                                                              ", DAGRUSOTA, DGRU, SDGRU, Mamba]")
parser.add_argument('-train_only', type=bool, required=False, default=False,
                    help='perform training on full input dataset without validation and testing')
parser.add_argument('-is_training', type=int, required=False, default=1, help='status')
parser.add_argument('-seq_len', type=int, default=256, help="时间窗口大小, window_size > pre_len")
parser.add_argument('-label_len', type=int, default=0, help='start token length')
parser.add_argument('-pre_len', type=int, default=336, help="预测未来数据长度")
parser.add_argument('-do_predict', action='store_true', help='whether to predict unseen future data')

# data
parser.add_argument('-shuffle', action='store_true', default=True, help="是否打乱数据加载器中的数据顺序")
parser.add_argument('--freq', type=str, default='d',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('-data', type=str, required=False, default='mianhua_multi', help='数据集名字')

# learning
parser.add_argument('-lr', type=float, default=0.01, help="学习率")
parser.add_argument('-drop_out', type=float, default=0.05, help="随机丢弃概率,防止过拟合")
parser.add_argument('-itr', type=int, default=5, help="实验次数")
parser.add_argument('-epochs', type=int, default=50, help="训练轮次")
parser.add_argument('-batch_size', type=int, default=16, help="批次大小")
parser.add_argument('-patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('-save_path', type=str, default='models')
parser.add_argument('-checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
# parser.add_argument('-num_workers', type=int, default=10, help='data loader num workers')

# TCN_models
parser.add_argument('-kernel_size', type=int, default=5)
parser.add_argument('-model_dim', type=list, default=[32, 64, 128], help='这个地方是这个TCN卷积的关键部分,它代表了TCN的层数我这里输'
                                                                         '入list中包含三个元素那么我的TCN就是三层，这个根据你的数据复杂度来设置'
                                                                         '层数越多对应数据越复杂但是不要超过5层')  # [64, 128, 256]
# DAGRU
parser.add_argument('-factor_x', type=int, default=4, help="时间序列分解次数")

# Attention
parser.add_argument('-n_heads', type=int, default=4, help="多头注意力头数")
parser.add_argument('-d_embed', type=int, default=256, help="特征维数映射key、value矩阵的维数")

# Linear
parser.add_argument('-individual', action='store_true', default=False,
                    help='DLinear: a linear layer for each variate(channel) individually')

# LSTM
parser.add_argument('-hidden_size', type=int, default=256, help="隐藏层单元数")
parser.add_argument('-num_layers', type=int, default=1)  # Mamba通用参数

# device
parser.add_argument('-use_gpu', type=bool, default=True)
parser.add_argument('-device', type=int, default=0, help="只设置最多支持单个gpu训练")

# option
parser.add_argument('-train', type=bool, default=True)
parser.add_argument('-test', type=bool, default=True)
parser.add_argument('-predict', type=bool, default=True)
parser.add_argument('-inspect_fit', type=bool, default=True)
parser.add_argument('-lr-scheduler', type=bool, default=True)

# 下面设置修改data名字后自动修改
parser.add_argument('-data_path', type=str, default='./data/processed_train_mutigai.csv', help="你的数据数据地址")
parser.add_argument('-target', type=str, default='value', help='你需要预测的特征列，这个值会最后保存在csv文件里')
parser.add_argument('-input_size', type=int, default=7, help='你的特征个数不算时间那一列7')
parser.add_argument('-features', type=str, default='MS', help='[M, S, MS],多元预测多元,单元预测单元,多元预测单元')

args = parser.parse_args()
# 指定实验设备
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

# 指定输出尺寸
if args.features == 'MS' or args.features == 'S':
    args.output_size = 1
else:
    args.output_size = args.input_size

data_parser = {
    'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'WTH': {'data': './data/WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
    'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
    'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1]},
    'stock': {'data': 'sh600000.csv', 'T': 'close', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'apple_stock_data': {'data': 'apple_stock_data.csv', 'T': 'Adj Close', 'M': [6, 6, 6], 'S': [1, 1, 1],
                         'MS': [6, 6, 1]},
    'Amazon': {'data': 'Amazon.csv', 'T': 'Adj Close', 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [6, 6, 1]},
    'hk_weather': {'data': 'hk_weather.csv', 'T': 'DTP', 'M': [4, 4, 4], 'S': [1, 1, 1], 'MS': [4, 4, 1]},
    'mianhua': {'data': 'mianhua.csv', 'T': 'value', 'M': [1, 1, 1], 'S': [1, 1, 1], 'MS': [1, 1, 1]},
    'mianhua_multi': {'data': './data/processed_train_mutigai.csv', 'T': 'value', 'M': [7, 7, 7], 'S': [1, 1, 1],
                      'MS': [7, 7, 1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.input_size = data_info[args.features][1]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    average_mse = []
    average_mae = []
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_ft{}_sl{}_pl{}_md{}_{}'.format(
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pre_len,
            args.model_dim, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        if not args.train_only:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mse, mae = exp.test(setting)
            average_mse.append(mse)
            average_mae.append(mae)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
    sorted_average_mse = sorted(average_mse)
    sorted_average_mae = sorted(average_mae)
    print(sum(sorted_average_mse[:5]) / len(average_mse[:5]), sum(sorted_average_mae[:5]) / len(average_mae[:5]))
else:
    ii = 4
    setting = '{}_{}_ft{}_sl{}_pl{}_md{}_{}'.format(
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pre_len,
        args.model_dim, ii)

    exp = Exp(args)  # set experiments

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
    torch.cuda.empty_cache()
