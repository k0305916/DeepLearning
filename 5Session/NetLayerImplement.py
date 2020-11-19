import numpy as np
from common.functions import softmax, cross_entropy_error

# ReLU层的作用就像电路中的开关一样。
# 正向传播时: 有电流通过的话，就将开关设为ON；没有电流通过的话，就将开关设为OFF。 
# 反向传播时:开关为ON的话，电流会直接通过；开关为OFF的话， 则不会有电流通过。
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask]=0
        dx = dout

        return dx
class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1+np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


# 神经网络的正向传播中进行的矩阵的乘积运算在几何学领域被称为“仿 A 射变换” 。因此，这里将进行仿射变换的处理实现为“Affine层”。
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)

        return dx

# Softmax层。考虑到这里也包含作为损失函数的交叉熵误 差（cross entropy error） ，所以称为“Softmax-with-Loss层”
# 使用交叉熵误差作为 softmax 函数的损失函数后，反向传播得到 （y 1 − t1 , y 2 − t2 , y 3 − t 3 ）这样“漂亮”的结果。实际上，这样“漂亮” 的结果并不是偶然的，而是为了得到这样的结果，特意设计了交叉 熵误差函数。回归问题中输出层使用“恒等函数”，损失函数使用 “平方和误差”，也是出于同样的理由（3.5节）。也就是说，使用“平 方和误差”作为“恒等函数”的损失函数，反向传播才能得到（y 1 t1, y 2− t2, y 3− t 3）这样“漂亮”的结果。
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # loss function
        self.y = None  # softmax output
        self.t = None # 监督数据(ont-hot vector)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx