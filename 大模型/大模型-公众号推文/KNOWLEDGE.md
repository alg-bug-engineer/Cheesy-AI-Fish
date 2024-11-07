### 0530：使用Sigmoid的逻辑回归为什么是线性分类器？
![](https://cdn.nlark.com/yuque/0/2024/png/406504/1717029991289-adf83417-fd8c-48e5-8661-a86db16654fc.png)

逻辑回归尽管使用了Sigmoid（逻辑）激活函数，但仍被称为线性分类器，原因在于它以线性方式模拟输入特征与二元目标变量的对数几率（logit）之间的关系。逻辑回归中的“线性”指的是它在特征空间中创建了一个线性决策边界，这是一个超平面。

  


在逻辑回归模型中，预测结果是通过将输入特征与权重相乘、加上偏置项，然后通过Sigmoid函数来计算的。Sigmoid函数的作用是将线性函数的输出压缩到(0, 1)区间内，以表示概率。但是，即使Sigmoid函数本身是非线性的，决策边界的线性性质是由输入特征和权重的线性组合决定的。这意味着，无论Sigmoid函数如何变换其输入，决策边界始终是线性的，因为它是由特征和权重的线性组合形成的。

  


换句话说，逻辑回归模型的输出是输入特征的线性函数的Sigmoid变换。即使Sigmoid变换将输出压缩到(0, 1)区间，用于分类的决策边界（即Sigmoid函数的倒置）本质上仍然是基于特征和权重的线性组合。

  


逻辑回归被称为线性分类器，是因为它的决策边界在特征空间中是线性的，而不是因为它不使用非线性函数。实际上，逻辑回归结合了线性模型和非线性激活函数来创建一个适合二分类问题的预测模型。

  


![](https://cdn.nlark.com/yuque/0/2024/png/406504/1717029991451-5d580551-c1a4-4f72-b5fa-efc1c703d96c.png)

### 大模型中token的核心技术：BPE算法
BPE（Byte Pair Encoding）算法是一种用于文本处理的字节对编码技术，广泛应用于自然语言处理（NLP）中的词汇表构建。BPE 是一种基于频率的迭代算法，用于将文本序列分解为更小的、更频繁出现的子单元，这些子单元可以是单个字符、字符对或其他符号。这种方法特别适用于处理词汇量大、出现频率低的单词，以及在处理多种语言时需要共享词汇表的情况。

BPE 算法的基本步骤如下：

1.  **初始化词汇表**：开始时，词汇表包含所有可能的单字符（例如，对于英文，这将是26个字母加上空格和其他标点符号）。 
2.  **合并规则**：算法会根据一定的规则（通常是合并频率最高的字符对）来迭代地扩展词汇表。在每次迭代中，算法会寻找并合并最常见的字符对，生成新的词汇表条目。 
3.  **重复迭代**：这个过程会重复进行，直到达到预定的词汇表大小或者合并规则不再有效为止。每次迭代后，新的词汇表条目会替换原来的字符对，成为文本中的新符号。 
4.  **生成分词规则**：最终，算法会生成一组分词规则，这些规则定义了如何将原始文本中的单词分解为词汇表中的子单元序列。 

BPE 算法的特点

+ **数据压缩**：BPE 算法可以有效地压缩文本数据，减少存储空间和计算资源的消耗。
+ **适应性强**：BPE 算法不依赖于特定的语言或符号集，可以灵活地应用于不同的文本处理任务。
+ **词汇共享**：在多语言模型中，BPE 算法可以帮助构建共享的词汇表，使得模型能够在不同的语言之间共享知识。

BPE 算法在自然语言处理领域有广泛的应用，尤其是在以下方面：

+ **机器翻译**：在构建多语言翻译系统时，BPE 可以帮助生成共享的词汇表，提高翻译效率。
+ **文本分类**：BPE 可以用于生成更紧凑的词汇表，提高文本分类模型的性能。
+ **词嵌入**：在构建词嵌入模型时，BPE 可以减少词汇表的大小，提高模型的训练和推理速度。

BPE 算法是一种简单而强大的文本预处理技术，它通过迭代合并字符对来优化词汇表，从而在不牺牲太多信息的情况下减少数据的复杂性。这种方法在处理大规模文本数据时尤其有用，能够显著提高自然语言处理任务的效率和效果。

### 大模型的int8量化
<font style="color:rgb(25, 27, 31);">Int8量化是一种优化技术，</font><font style="color:#601BDE;">它将32位的浮点数（float32）模型参数转换为8位整数（int8）</font><font style="color:rgb(25, 27, 31);">。这有助于减少模型的内存占用，加快其执行速度，并有可能减少能耗，特别是在边缘设备上。以下是完成Int8量化所采用的基本计算公式：</font>

<font style="color:#2F4BDA;">通用 INT8-Bit量化</font>

**（一）公式：**

<font style="color:rgb(25, 27, 31);">量化和反量化过程如下所示：</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707188640453-34857517-9ae0-4602-b6f5-e33ce58fdde0.png)

其中 c 是量化常数

**（二）通用流程：**

**<font style="color:rgb(25, 27, 31);">（1）确定缩放因子（Scale）和零点（Zero Point）：</font>**

<font style="color:rgb(25, 27, 31);">首先，计算缩放因子（scale）和零点（zero point）。</font><font style="color:#601BDE;">缩放因子用于将浮点数映射到整数范围，零点则确保浮点数0可以被准确地表示为整数</font><font style="color:rgb(25, 27, 31);">。</font>

<font style="color:rgb(25, 27, 31);">计算scale和zero_point通常需要最大值（max_val，127）和最小值（min_val，-128）：</font>

```python
scale = (max_val - min_val) / (2^7 - 1)  
# 127是int8的最大值 
zero_point = round(-min_val / scale)
```

<font style="color:rgb(25, 27, 31);">这里，zero_point需要在-128到127的范围内，因为这是8位有符号整数的范围。</font>

**<font style="color:rgb(25, 27, 31);">（2）应用量化：</font>**

<font style="color:#601BDE;">使用scale和zero_point，量化一个浮点数（float_val）</font><font style="color:rgb(25, 27, 31);">成一个8位整数（int_val）：</font>

```python
int_val = round(float_val / scale) + zero_point
```

**<font style="color:rgb(25, 27, 31);">（3）整数裁剪：</font>**

<font style="color:rgb(25, 27, 31);">如果量化后的整数超出了-128到127的范围，它需要被裁剪以保持在有效范围内。</font>

```python
int_val = min(127, max(-128, int_val))
```

<font style="color:rgb(25, 27, 31);">这样，</font><font style="color:#601BDE;">量化过程就可以将32位浮点数数组转换为能够在8位整数范围内表示的数组</font><font style="color:rgb(25, 27, 31);">，这减少了内存占用，同时可能提高了执行速度。</font>

### int8量化的优化：LLM.int8()
**<font style="color:rgb(25, 27, 31);">方法：使用混合精度分解（Mixed-precision Decomposition）</font>**<font style="color:rgb(25, 27, 31);">：</font><font style="color:#601BDE;">异常值会影响量化后模型的计算精度</font><font style="color:rgb(25, 27, 31);">。transformers存在一些非常大的异常值，在数组中加入一个</font><font style="color:#601BDE;">异常值98</font><font style="color:rgb(25, 27, 31);">，处理后的结果为[-0. , -0. , 0. , -0. , -0. , -0.77, -2.31, 0. , -0.77, 98. ]，</font><font style="color:#601BDE;">转化后的大部分信息都丢失了</font><font style="color:rgb(25, 27, 31);">。因此，</font><font style="color:#601BDE;">可</font><font style="color:#DF2A3F;">以将异常特征的从矩阵中分离出来，保留原始精度进行矩阵乘法，剩余的（99.9%）特征进行int8量化计算</font><font style="color:rgb(25, 27, 31);">。</font><font style="color:#601BDE;">然后将两者的输出结合在一起，从而保留模型性能</font><font style="color:rgb(25, 27, 31);">。</font>

**<font style="color:rgb(248, 57, 41);">LLM.int8()会通过以下三步完成矩阵乘法：</font>**

1. <font style="color:rgb(25, 27, 31);">从矩阵隐藏层中，以列为单位，抽取outliers（数值超出全局阈值范围的）。</font>
2. <font style="color:rgb(25, 27, 31);">分别通过</font>**<font style="color:rgb(25, 27, 31);">FP16精度</font>**<font style="color:rgb(25, 27, 31);">对</font>**<font style="color:rgb(25, 27, 31);">outliers</font>**<font style="color:rgb(25, 27, 31);">的部分做矩阵乘法，通过量化</font>**<font style="color:rgb(25, 27, 31);">int8精度</font>**<font style="color:rgb(25, 27, 31);">对其他的做矩阵乘法。</font>
3. <font style="color:rgb(25, 27, 31);">将量化的部分恢复成FP16，然后将两部分合在一起。</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707107178748-ae472a21-f12d-4691-9f8d-0e8799728b12.png?x-oss-process=image%2Fresize%2Cw_770%2Climit_0)

![](https://cdn.nlark.com/yuque/0/2024/gif/35381469/1707107288358-6d08a6b8-5af3-47a8-81bb-60d26a3afa81.gif)

