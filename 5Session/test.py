import numpy as np



X = np.random.rand(2) # 1row 2column
W = np.random.rand(2,3) # 2row 3column
B = np.random.rand(3) # 1row 3column


print(X.shape)
print(W.shape)
print(B.shape)

Y = np.dot(X,W)+B

print(Y)