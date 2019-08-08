#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


#创建数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

#将tensor添加到Variable 中# 画图
x, y = torch.autograd.Variable(x), Variable(y)

plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

#构建神经网络
# torch 中的体系. 先定义所有的层属性(__init__()),
# 然后再一层层搭建(forward(x))层于层的关系链接.
# 建立关系的时候, 会用到激励函数
import torch.nn.functional as F #引入激励函数

#继承torch 的 Module
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_ouput):
        #继承__init__ 功能
        super(Net, self).__init__()
        #定义每层的样式
        #隐藏层的线性输出
        self.hidden  = torch.nn.Linear(n_feature, n_hidden)
        #输出层的线性输出
        self.predict = torch.nn.Linear(n_hidden, n_ouput)

    #定义forward功能,同时也是Module中forward功能
    def forward(self, x):
        #正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))
        #输出
        x = self.predict(x)
        return x

net = Net(n_feature=1, n_hidden=10, n_ouput=1)

#显示net结构
print net
"""
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
"""

#训练网络
#optimizer是训练的工具, 选用随机梯度下降
#传入net的所有参数, 学习率
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
#试用均方差做损失函数
loss_func = torch.nn.MSELoss()

#训练
#使plot可以循环打印
plt.ion()

for t in range(1000):
    prediction = net(x)     # 输入 x 得到 x的预测值

    loss = loss_func(prediction, y)

    optimizer.zero_grad()   # 为下一轮训练迭代清空梯度值
    loss.backward()         # 反向传播计算梯度
    optimizer.step()        # 使用梯度

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
