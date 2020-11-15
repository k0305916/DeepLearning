**神经网络的学习**
为了使神经网络能进行学习，将导入**损失函数**这一指标，而学习的目的就是以该损失函数为基准，找出能使它的值达到最小的权重参数，故使用**函数斜率的梯度法**。
# 方案：
* 图像提取+机器学习
  1. 从图像中提取特征量；
     * 特征量：可以从输入数据中准确提取重要的数据的转换器。
  2. 利用机器学习技术学习这些特征量。
     * CV领域中，特征量包括：SIFT，SURF，HOG等；
     * SVM，KNN等分类器进行学习。
  3. 该转换器仍是由人工设计，因此针对不同的问题，需考虑合适的转换器。
* 神经网络
  * 优点为：对所有的问题都可以用同样的流程来解决。 
  * 损失函数，可以使用任意函数，但一般为*均方误差(MSE)*和*交叉熵误差*等；
# 神经网络的学习
## 均方误差
Description: $y_k: 神经网络的输出，t_k：监督数据；k：数据的维数。$
$$E=\frac{1}{2}\sum_k(y_k - t_k)^2$$
## 交叉熵误差
Description: $log: 以e为底的自然对数log; y_k: 神经网络的输出； t_k：正确解标签$
$$E=-\sum_k{t_k * \log{y_k}}$$
## mini-batch 学习
神经网络的学习也是从训练数据中选出一批数据（称为mini-batch，小批量），然后对每个mini-batch进行学习。
# 学习算法的实现
## Steps:
1. mini-batch；
    从训练数据中随机选出一部分数据，这部分数据称为mini-batch。我们的目标是减小mini-batch的损失函数的值；
2. calc gradient;
    为了减小mini-batch的损失函数的值，需要求出各个weight para的梯度。
    梯度表示损失函数的值减小最多的方向
3. update the parameters;
   将weight para沿gradient的方向进行微小的update。
4. repeat 2~3
   repeat until eplison or iterate的次数。