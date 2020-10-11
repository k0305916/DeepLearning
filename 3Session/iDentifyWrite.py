import sys, os
#为了导入父目录中的文件而进行的设定
o_path = os.getcwd()
sys.path.append(o_path)
print(o_path)
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
import numpy as np
from PIL import Image
import pickle


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = \
            load_mnist(flatten=True, normalize=True,one_hot_label=False)
    return x_test, t_test

# 因为之前我们假设学习已经完成，所以学习到的参数被保存下来。
# 假设保存在 sample_weight.pkl 文件中，在推理阶段，我们直接加载这些已经学习到的参数。
def init_network():
    with open('./3Session/sample_weight.pkl','rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

# #  show the img
# # -----------------------------------
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# img=x_train[0]
# label = t_train[0]
# print(img.shape)# 一维
# img = img.reshape(28,28) #把图像的形状变成原来的尺寸
# print(img.shape)# 二维

# # show the img
# img_show(img)

# # # 输出各个数据的形状
# # print(x_train.shape)
# # -----------------------------------


# # single processing
# # -----------------------------------
# x, t = get_data()
# network = init_network()

# accuracy_cnt = 0
# for i in range(len(x)):
#     y = predict(network, x[i])
#     p = np.argmax(y) # 获取概率最高的元素的索引
#     if p == t[i]:
#         accuracy_cnt += 1

# print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
# # -----------------------------------

# Multiple Processing
# -----------------------------------
x, t = get_data()
network = init_network()

batch_size = 100  # 批数量 
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
    
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
# -----------------------------------








