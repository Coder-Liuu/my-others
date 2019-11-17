from time import time
import numpy as np
import keras
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation,Conv1D,Flatten,MaxPool1D
from sklearn.metrics import mean_squared_error

# 0.设置开始时间
start = time()

# 1.加载数据集
train= pd.read_csv('./data/data.csv',header=None)   # 可设置nrows=100 来设置只读取数据的前100行
# 将决策属性和数据分开
x_data = train[list(range(16))]
y_data = train[16].values
# 将输入的数据标准化
std = StandardScaler()
x = std.fit_transform(x_data)
# 划分数据集
x_train,x_test,y_train,y_test = train_test_split(x,y_data,test_size=0.2)
# 将数据升维度，以便卷积  (100,16)  ---->  (100,16,1)
x_train = x_train.reshape(len(x_train),16,1)
x_test = x_test.reshape(len(x_test),16,1)


# 2.开始堆叠个个神经层
model = Sequential()
# 堆叠一个卷积层 30个过滤器 过滤器的尺寸为3   输入数据的形状是(16,1)   激活函数采用relu  
# 池化前数据的形状 16,1   -----> 卷积后 14,1
model.add(Conv1D(filters=30,kernel_size=3, input_shape=(16,1), activation="relu") )
# 定义最大化池化层  ---> 可以加快速度,但是会损失一些信息
# model.add(MaxPool1D(pool_size=2))
# 将数据维度变成只有一个维度的数据
model.add(Flatten())
# 在搭建一个隐藏层
model.add(Dense(8*30))
# 中间使用 sigmoid，作为激活函数
model.add(Activation('sigmoid'))
# 输出层一个神经元
model.add(Dense(1))
# 误差函数选择 mse  优化器使用adam
model.compile(loss='mse',
              optimizer='adam',
            )
# 模型的训练  batch_size:每批训练的大小   epochs:迭代多少次
loss_and_metrics = model.fit(x_train, y_train, batch_size=32,epochs=30)


# 3.使用模型来预测数据
y_pred = model.predict(x_test)
print("实际的前十个数据是:",y_test[:10])
print("预测的前十个数据是",y_pred[:10])
# 计算MSE误差
loss = mean_squared_error(y_test,y_pred)
print("MSE误差是",loss)

# 设置结束时间
end = time()

print("一共用时",end-start)
