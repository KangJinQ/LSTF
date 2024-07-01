import math

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'mianhua': Dataset_Custom,
    'mianhua_multi': Dataset_Custom,
    'WTH': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    train_only = args.train_only

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pre_len],
        features=args.features,
        target=args.target,
        train_only=train_only
    )
    print(flag, len(data_set), "batches_nums:", math.floor(len(data_set)/args.batch_size))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        # num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
