import os
import time

from matplotlib import pyplot as plt

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import TCN, Linear, DLinear, LSTM, GRU, DAGRU, DAGRUSOTA, DGRU, SDGRU, Mamba
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.tools import plot_loss_data, visual, EarlyStopping, adjust_learning_rate

# have_attention = True
have_attention = False

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'TCN': TCN,
            'Linear': Linear,
            'DLinear': DLinear,
            'LSTM': LSTM,
            'GRU': GRU,
            'DAGRU': DAGRU,
            'DAGRUSOTA': DAGRUSOTA,
            'DGRU': DGRU,
            'SDGRU': SDGRU,
            'Mamba': Mamba,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        # with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()

            if have_attention:
                pred_y, attn_g = self.model(batch_x)
            else:
                pred_y = self.model(batch_x)

            f_dim = -1 if self.args.features == 'MS' else 0
            pred_y = pred_y[:, -self.args.pre_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pre_len:, f_dim:].to(self.device)

            loss = criterion(pred_y.detach().cpu(), batch_y.detach().cpu())
            total_loss.append(loss)

        self.model.train()
        return sum(total_loss) / len(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        checkpoints_path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)

        start_time = time.time()  # 计算起始时间
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        criterion = nn.MSELoss()

        self.model.train()

        results_loss = []
        for i in range(self.args.epochs):
            losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                optimizer.zero_grad()

                if have_attention:
                    y_pred, attn_g = self.model(batch_x)
                else:
                    y_pred = self.model(batch_x)
                # 判断目标特征个数
                f_dim = -1 if self.args.features == 'MS' else 0
                y_pred = y_pred[:, -self.args.pre_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pre_len:, f_dim:].to(self.device)

                single_loss = criterion(y_pred, batch_y)
                single_loss.backward()
                optimizer.step()
                losses.append(single_loss.detach().cpu().numpy())

            if not self.args.train_only:
                # 验证集 valid
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)
                tqdm.write(
                    f"\t Epoch {i + 1} / {self.args.epochs}, Loss: {(sum(losses) / len(losses)):.6f},"
                    f" Vali_Loss: {vali_loss:.6f}, Test_Loss: {test_loss:.6f}")
            else:
                tqdm.write(f"\t Epoch {i + 1} / {self.args.epochs}, Loss: {(sum(losses) / len(losses)):.6f}")
            results_loss.append(sum(losses) / len(losses))

            # 早停
            early_stopping(vali_loss, self.model, checkpoints_path)  # vali_loss

            if early_stopping.early_stop:
                print("Early stopping")
                break
            # 动态调整学习率
            adjust_learning_rate(optimizer, i + 1, self.args)
            # torch.save(self.model.state_dict(), f'./saved_models/{setting}.pth')

        best_model_path = checkpoints_path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        print(f">>>>>>>>>>>>>>>>>>>>>>模型已保存,用时:{(time.time() - start_time) / 60:.4f} min<<<<<<<<<<<<<<<<<<")
        plot_loss_data(results_loss)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        checkpoints_path = os.path.join(self.args.checkpoints, setting)

        if test:
            print('loading model')
            # self.model.load_state_dict(torch.load(os.path.join('./saved_models/' + setting, '.pth')))
            self.model.load_state_dict(torch.load(os.path.join(checkpoints_path, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        qr_path = './qualitative_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for bn, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if have_attention:
                    y_pred, attn_g = self.model(batch_x)
                else:
                    y_pred = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                y_pred = y_pred[:, -self.args.pre_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pre_len:, f_dim:].to(self.device)

                y_pred = y_pred.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                # for i in range(len(y_pred)):
                #     preds.append(y_pred[i][:, -1])  # 单变量
                #     trues.append(batch_y[i][:, -1])
                preds.append(y_pred)
                trues.append(batch_y)
                inputx.append(batch_x.detach().cpu().numpy())

                if bn % 5 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], y_pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(bn) + '.pdf'))
                    if test:
                        np.save('./qualitative_results/predictions' + setting + str(bn) + '.npy', pd)
                        np.save('./qualitative_results/groundTruth256' + str(bn) + '.npy', gt)

        if have_attention:
            plt.imshow(attn_g[0, 0, :, :].cpu().detach().numpy(), cmap='hot', interpolation='nearest')
            plt.title('Attention Matrix')
            plt.xlabel('Source Sequence')
            plt.ylabel('Target Sequence')
            plt.colorbar()  # 添加颜色条
            # 保存热力图为图像文件
            plt.savefig('heatmap.png')

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mse, mae, rmse, mape, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, corr:{}'.format(mse, mae, rmse, mape, corr))
        f.write('\n')
        f.write('\n')
        f.close()
        np.save(folder_path + 'pred.npy', preds)
        return mse, mae

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join('./saved_models')
            best_model_path = path + '/' + setting + '.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in pred_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # pred = self.model(batch_x)

                if have_attention:
                    pred, attn_g = self.model(batch_x)
                else:
                    pred = self.model(batch_x)

                pred = pred.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.concatenate(preds, axis=0)
        if (pred_data.scale):
            preds = pred_data.inverse_transform(preds)

        # result save
        folder_path = './pred_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # np.save(folder_path + 'real_prediction.npy', preds)
        # pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1),
        #              columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)

        return
