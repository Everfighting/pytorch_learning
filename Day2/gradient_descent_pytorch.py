from __future__ import print_function
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np


X=Variable(torch.randn(100,1),requires_grad=False)
Y=3*X+5

w=Variable(torch.randn(1,1),requires_grad=True)
b=Variable(torch.randn(1,1),requires_grad=True)

def gradient_descent():
    for i in range(100):
        y_ = X @ w + b
        loss=torch.mean((Y-y_)**2)
        loss.backward()
        print ("w.grad: ",w.grad)
        w.data = w.data - 0.1 * w.grad.data
        b.data = b.data - 0.1 * b.grad.data
        w.grad.data.zero_()
        b.grad.data.zero_()
        print ("w is :%3.2f , b is :%3.2f, loss is :%3.2f"%(w.mean(),b.mean(),loss))

def stochastic_gradient_descent():
    for i in range(100):
        for x,y in zip(X,Y):
            y_ = x @ w + b
            loss = torch.mean((y - y_) ** 2)
            loss.backward()
            print("w.grad: ", w.grad)
            w.data = w.data - 0.1 * w.grad.data
            b.data = b.data - 0.1 * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()
            print("w is :%3.2f , b is :%3.2f, loss is :%3.2f" % (w.mean(), b.mean(), loss))

