import numpy as np
import sys, os
from dataset.mnist import load_mnist

def mse_Func(y,t):
    return 0.5 * np.sum((y-t)**2)

# # single cross entroy error
# def cross_entroy_error(y,t):
#     delta = 1e-7
#     return -np.sum(t * np.log(y + delta))

# cross entroy error with min-batch
def cross_entroy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    # one-hot style
    return -np.sum(t*np.log(y + 1e-7)) / batch_size

    # label style
    # return -np.sum(np.log(y[np.arange(batch_size),t]+1e-7)) / batch_size

# numerical differentiation(数值微分)
def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def numeriacal_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x) #生成和x形状相同的数组

    for idx in range(x.size):
        tmp_val = x[idx]

        #f(x+h)的计算
        x[idx]=tmp_val+h
        fxh1 = f(x)

        #f(x-h)的计算
        x[idx]=tmp_val-h
        fxh2 = f(x)

        grad[idx]=(fxh1-fxh2) / (2*h)
        x[idx]=tmp_val  #还原值

    return grad

# 参数 f 是要进行最优化的函数， init_x 是初始值， lr 是学习率learning rate， step_num 是梯度法的重复次数。
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numeriacal_gradient(f,x)
        x -= lr * grad

    return x    


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
