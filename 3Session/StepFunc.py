import numpy as np
import matplotlib.pylab as plt

# h(x) = 0 if x <= 0
# h(x) = 1 if x > 0
def step_func(x):
    return np.array(x > 0, dtype=np.int)


x = np.arange(-5.0, 5.0, 0.1)
y = step_func(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # 指定Y轴的范围
plt.show()
