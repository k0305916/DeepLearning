import numpy as np
import matplotlib.pylab as plt

# h(x) = 0 if x <= 0
# h(x) = x if x > 0
def ReLU(x):
    return np.maximum(0,x)


x = np.arange(-5.0, 5.0, 0.1)
y = ReLU(x)
plt.plot(x, y)
plt.show()
