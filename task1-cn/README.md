### 实现基于logistic/softmax regression的文本分类

1. 参考
   1. 《[神经网络与深度学习](https://nndl.github.io/)》 第2/3章
2. 数据集：外卖评论情感分类数据集
3. 实现要求：NumPy
4. 需要了解的知识点：

   1. 文本特征表示：Bag-of-Word，N-gram
   2. 分类器：logistic/softmax  regression，损失函数、（随机）梯度下降、特征选择
   3. 数据集：训练集/验证集/测试集的划分
   4. 中文分词(jieba, pkuseg)

5. 实验：
   1. 分析不同的特征、损失函数、学习率对最终分类性能的影响
   2. shuffle 、batch、mini-batch 
6. 时间：一周