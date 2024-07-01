import pandas as pd

# 读取Excel文件
df = pd.read_excel("train_mutigai.xlsx")

# 对于input列，用上一个出现的值填充之前的所有空值
df['input'] = df['input'].ffill()

# 1. 截取2002-06-10之后的数据
df = df[df['date'] >= '2004-07-15']

# 2. 删除value列空缺的行
df = df.dropna(subset=['value'])

# 删除output列
df.drop(columns=['output'], inplace=True)

# 3. 使用空缺值前后的数值的平均作为替代
cols_to_interpolate = ['USD', 'EUR', 'GBP', 'Dtd', 'WTI']
for col in cols_to_interpolate:
    df[col] = df[col].fillna(df[col].interpolate())



# 保存处理后的数据为CSV文件
df.to_csv("processed_train_mutigai.csv", index=False)
