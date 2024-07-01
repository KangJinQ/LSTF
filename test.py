import torch
import torch.nn as nn

# 固定随机种子以保证结果可复现性
torch.manual_seed(42)


# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(in_features=7, out_features=4)

    def forward(self, x):
        return self.linear(x)


# 创建模型实例和MSE loss
model = MyModel()
criterion = nn.MSELoss()

# 初始化两个矩阵
input1 = torch.randn(1, 7)
input2 = torch.randn(4, 7)

y = torch.arange(1, 5).float()
print(y)

input = input1

# 训练两轮
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(20):
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output.permute(1, 0)[:, -1], y)  # 只取最后一列进行反向传播
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch + 1}/2], Loss: {loss.item()}')

print(model(input))

# 打印训练后模型的权重
print("Model's learned weights:")
print(model.linear.weight)
