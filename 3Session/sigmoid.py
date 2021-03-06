import numpy as np
import matplotlib.pylab as plt

# h(x) = 1 / (1+exp(-x))
# exp(-x) = e^(-x) & e = 2.7182....
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)
y = Sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # 指定Y轴的范围
plt.show()
