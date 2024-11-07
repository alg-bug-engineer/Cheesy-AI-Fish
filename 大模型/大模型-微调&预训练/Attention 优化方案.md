通过对一些训练好的Transformer 模型中的注意力矩阵进行分析发现，其中很多通常是稀疏的，因此可以通过限制Query-Key 对的数量来减少计算复杂度。这类方法就称为稀疏注意力（Sparse

Attention）机制。可以将稀疏化方法进一步分成两类：基于位置信息和基于内容。

基于位置的稀疏注意力机制的基本类型如图所示，主要包含如下五种类型：

1. 全局注意力（Global Attention）：为了增强模型建模长距离依赖关系，可以加入一些全局节点；
2. 带状注意力（Band Attention）：大部分数据都带有局部性，限制Query 只与相邻的几个节点进行交互；
3. 膨胀注意力（Dilated Attention）；与CNN 中的Dilated Conv 类似，通过增加空隙以获取更大的感受野；
4. 随机注意力（Random Attention）：通过随机采样，提升非局部的交互；
5. 局部块注意力（Block Local Attention）：使用多个不重叠的块（Block）来限制信息交互。

![https://cdn.nlark.com/yuque/0/2024/png/406504/1716426853807-bbdb17e3-26a4-44c0-a5d2-2dbfbc4767a4.png](https://cdn.nlark.com/yuque/0/2024/png/406504/1716426853807-bbdb17e3-26a4-44c0-a5d2-2dbfbc4767a4.png)

基于内容的稀疏注意力是是根据输入数据来创建稀疏注意力，其中一种很简单的方法是选择和给定查询（Query）有很高相似度的键（Key）。Routing Transformer 采用K-means 聚类方法，针对

![https://cdn.nlark.com/yuque/0/2024/png/406504/1716426950698-494ffa46-ede6-403f-a2a9-d990350bb615.png](https://cdn.nlark.com/yuque/0/2024/png/406504/1716426950698-494ffa46-ede6-403f-a2a9-d990350bb615.png)

和

![https://cdn.nlark.com/yuque/0/2024/png/406504/1716426951012-5138c074-e0c0-4324-be7d-c27978ea81ae.png](https://cdn.nlark.com/yuque/0/2024/png/406504/1716426951012-5138c074-e0c0-4324-be7d-c27978ea81ae.png)

一起进行聚类，类中心向量集合为

![https://cdn.nlark.com/yuque/0/2024/png/406504/1716426950861-47fb391d-126b-42b3-8a46-a73170db5c1b.png](https://cdn.nlark.com/yuque/0/2024/png/406504/1716426950861-47fb391d-126b-42b3-8a46-a73170db5c1b.png)

，其中k 是类中心个数。每个Query 只与其处在相同簇（Cluster）下的Key 进行交互。中心向量采用滑动平均的方法进行更新：

![https://cdn.nlark.com/yuque/0/2024/png/406504/1716426950722-339b94b4-fa11-4e29-9546-9215cad0d1ff.png](https://cdn.nlark.com/yuque/0/2024/png/406504/1716426950722-339b94b4-fa11-4e29-9546-9215cad0d1ff.png)