import numpy as np
import matplotlib.pylab as plt

# 存在缺陷：会存在溢出的问题
def softmax_func(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def softmax_opt_func(a):
    c = np.max(a)
    exp_a = np.exp(a-c) # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


x = np.arange(0.3, 2.9, 4.0)
y = softmax_opt_func(x)
print(y)
