# 前提
# 神经网络中有合适的权重和偏置，调整权重和偏置以便拟合训练数据的 过程称为学习。神经网络的学习分为下面4个步骤。
# 步骤1（mini-batch）
# 从训练数据中随机选择一部分数据。
# 步骤2（计算梯度）---误差反向传播法
# 计算损失函数关于各个权重参数的梯度。
# 步骤3（更新参数）
# 将权重参数沿梯度方向进行微小的更新。
# 步骤4（重复）
# 重复步骤1、步骤2、步骤3。


# Two Layer Net Implement
import sys, os
import numpy as np
from common.layer import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                  weight_init_std=0.01):
        # initial weight
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        OrderedDict 是有序字典，“有序”是

        # 记住向字典里添加元素的顺序。
        # 因此，神经网络的正向传播只需按照添加元 素的顺序调用各层的 forward() 方法就可以完成处理，
        # 而反向传播只需要按 照相反的顺序调用各层即可。
        # 因为Affine层和ReLU层的内部会正确处理正向传播和反向传播，
        # 所以这里要做的事情仅仅是以正确的顺序连接各层，再按顺序（或者逆序）调用各层。

        # create the layer
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        
        self.lastlayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    #  x: input data; t: supervise data;
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis = 1)
        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy

    # x:input data; t: supervise data
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastlayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        # setting
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads