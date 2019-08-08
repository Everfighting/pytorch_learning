# 什么是Pytorch，为什么选择Pytroch？
  Pytorch是基于python的一个科学计算工具，适用于以下情况：
  - 在使用GPUs的情况下，作为替代numpy的一个工具
  - 是一个深度学习开发平台，提供了最大程度的灵活性和速度

# Pytroch的安装
  - 安装blog https://deeplizard.com/learn/video/UWlFM0R_x6I
    
# 配置Python环境
# 准备Python管理器
# 通过命令行安装PyTorch

```
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```

# PyTorch基础概念

## 一般典型的神经网络训练步骤如下：
  - 定义带学习参数 (learnable parameters or weights) 的神经网络；
  - 输入数据集进行迭代计算；
  - 使用神经网络处理输入；
  - 计算损失；
  - 将梯度返回到网络的参数中；
  - 更新参数。

## Pytorch基本概念

    import torch
    import numpy as np 
    import pandas as pd 
    from torch import nn 
    from torch.autograd import Variable

## Tensor张量

表示的是一个多维矩阵，零维就是一个点，一维就是向量，二维就是一般的矩阵，多维就相当于一个多维矩阵，Tensor可以和numpy的ndarray相互对应，Tensor可以和ndarray相互转换。


## Variable 变量

variable 提供了自动求导功能，Variable和Tensor没有本质区别，不过Variable会被放入一个计算图中，然后进行前向传播，反向传播，自动求导。

Variable有三个比较重要的组成属性：data， grad， grad_fn

data： 可以取出Variable中的Tensor数值

grad_fn： 表示的是得到这个Variable的操作，比如加减或者乘除

grad： 是这个Variable的反向传播梯度


## Dataset 数据集
torch.utils.data.Dataset 代表这一数据的抽象类吗可以自己定义数据类的继承和重写，只需要定义_len_和 _getitem_ 两个函数。


## nn.Module 模组
PyTorch 编写的神经网络，所有层结构和损失函数都来自于torch.nn，所有的模型构建都从这个基类nn.Module继承。


## torch.optim 优化
在机器学习或者深度学习中，我们需要通过修改参数使得损失函数最小化（或最大化），优化算法就是一种调整模型参数更新的策略。

模型的保存和加载
两种保存方式：

保存整个模型的结构信息和参数信息，保存的对象是模型model
保存模型的参数，保存的对象是模型的状态model.state_dict()
两种加载方式：

加载完整的模型结构和参数信息，网络较大的时候记载时间比较长，同时存储空间也比较大
加载模型参数信息，需要先导入模型的结构，然后再导入模型


# 通用代码实现流程(实现一个深度学习的代码流程)
