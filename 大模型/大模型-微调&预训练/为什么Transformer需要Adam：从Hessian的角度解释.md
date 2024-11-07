## 1. 引言
Transformer模型已成为AI发展的主要驱动力。然而，对Transformer训练的理解仍然有限。一个引人注目的现象是，Transformer的训练在很大程度上依赖于Adam优化器，相比之下，随机梯度下降与动量（SGD）在Transformer上的表现明显不如Adam（例如，见图3）。但是，造成这种性能差距的原因仍不清楚。

本文通过Hessian矩阵的视角探索了为什么SGD在Transformer上表现不如Adam。研究从调查Transformer的完整Hessian谱开始，即Hessian的完整特征值密度（见图1）。理论上，完整的Hessian谱在很大程度上决定了基于梯度的方法的行为。

![](https://cdn.nlark.com/yuque/0/2024/png/406504/1728716834661-45e314ff-945c-4b4a-b90b-9dcae9c0a44d.png)

作者使用数值线性代数工具，比较了CNNs（SGD与Adam表现相当）和Transformers（SGD明显落后于Adam）的完整谱。然而，如图1所示，尽管优化器行为不同，CNNs和Transformers的谱通常非常相似。因此，作者未能在完整的Hessian谱中识别出与Transformer上Adam和SGD之间差距相关的关键特征。为了揭示原因，需要对Hessian进行更细粒度的研究。

通过分解CNNs和Transformers的结构，作者注意到CNNs是由相似参数块（卷积层）的重复堆叠构成，而Transformers涉及非顺序堆叠的不同参数块（例如，注意力中的Query、Key、Value块和MLP层）。

基于这些观察，作者提出了一种可能的解释：Transformer固有的"异质性"。他们提供了经验和理论证据来支持这一解释。主要贡献可以总结如下：

1. 解释了为什么SGD在Transformer上表现不如Adam，通过检查块wise Hessian谱。
    1. 识别了一种称为"块异质性"的现象，指不同参数块之间的Hessian谱的巨大差异。
    2. 验证了块异质性阻碍了SGD的性能。
2. 在二次模型上的理论结果，构建了有和没有块异质性的凸二次问题，发现梯度下降（GD）在具有块异质性的问题上明显落后于Adam。

## 2. 主要结果
### 2.1 问题设置
作者引入了以下符号：

+ $ L(w) $：训练损失
+ $ w \in \mathbb{R}^d $：神经网络参数
+ $ \nabla L(w) \in \mathbb{R}^d $：训练损失相对于神经网络参数的梯度
+ $ \nabla^2 L(w) \in \mathbb{R}^{d \times d} $：训练损失的Hessian矩阵
+ $ [d] $：表示索引集 $ \{1, 2, \cdots, d\} $
+ $ \{D_l\}_{l=1}^L $：$ [d] $ 上的任意划分，其中 $ d_l \triangleq |D_l| $
+ $ \{w_l\}_{l=1}^L $：$ w $ 被划分成的 $ L $ 个参数块，其中 $ w_l = \mathbb{R}^{d_l} $ 由索引在第 $ l $ 个块 $ D_l $ 中的参数组成
+ $ [\nabla^2 L(w)]_l \in \mathbb{R}^{d_l \times d_l} $：$ l $ 参数块 $ w_l $ 的Hessian，其中 $ [\nabla^2 L(w)]_{l,i,j} = \frac{\partial^2}{\partial w_{l,i} \partial w_{l,j}}L(w_l) $

注意，$ [\nabla^2 L(w)]_l $ 是 $ \nabla^2 L(w) $ 的第 $ l $ 个主对角块子矩阵。

### 2.2 完整Hessian谱的信息不足
作者首先研究了Transformer的完整Hessian谱，原因有二：

1. Hessian谱显著影响基于梯度的方法的行为。
2. 先前研究表明，Hessian谱提供了对神经网络现象的洞察，如BatchNorm对训练速度的影响。

作者比较了CNNs（SGD与Adam表现相当）和Transformers（SGD明显落后于Adam）的完整谱，如图1所示。然而，结果表明，完整的Hessian谱可能不足以解释Adam和SGD在Transformer上的性能差距。作者详细分析了以下方面：

+ (A) 谱的离散程度：观察到不同模型的特征值离散程度相似，Transformers没有明显的大离群值。
+ (B) 谱的形状：对于所有CNNs和Transformers，在初始化时谱的形状大致对称分布在0周围。
+ (C) 训练过程中谱的演化：随着训练进行，大多数负特征值消失，形状演变为"主体"和一些"离群值"的组合。

由于CNNs和Transformers的谱形状和演化相当相似，这些特征无法解释为什么SGD在Transformer上表现较差。因此，需要对Hessian进行更细致的研究。

### 2.3 通过块wise Hessian谱的主要发现
![](https://cdn.nlark.com/yuque/0/2024/png/406504/1728717043713-19ad0f02-7ebf-4ad6-bafb-88b2c11ccce9.png)

作者发现了一些在完整Hessian谱分析中被忽视的关键特征：

1. Hessian结构：现有文献表明，MLPs的Hessian接近块对角矩阵。作者在小型Transformers中也观察到近似块对角的Hessian，如图2所示。
2. Transformer的构建规则：CNNs由相似参数块（卷积层）的重复堆叠构成，而Transformers包含非顺序堆叠的不同参数块（如注意力中的Query、Key、Value和MLP层）。

基于这些观察，作者假设块wise Hessian谱，即Hessian主对角块 $ [\nabla^2 L(w)]_l, l \in [L] $ 的谱，可能提供额外的洞察。

作者展示了VGG16（CNN）和BERT（Transformer）的块wise谱形状，如图3所示。在BERT中，嵌入层、注意力层和MLP块的Hessian谱差异很大。相比之下，在ResNet中，卷积层的谱相似。作者进一步计算了所有可能的块对之间特征值密度的Jensen-Shannon（JS）距离，结果如图4所示。

![](https://cdn.nlark.com/yuque/0/2024/png/406504/1728717081640-a3f2a0e0-7460-45dc-8ab3-c136b82e75ee.png)

这些结果揭示了一个新现象：在所有检查的Transformers中，块wise Hessian谱在不同块之间差异很大。作者将这种现象称为"块异质性"。相比之下，CNNs的块wise Hessian谱相似，没有观察到块异质性。这表明块异质性是区分CNNs和Transformers的一个重要特征。

### 2.4 SGD在具有块异质性的各种任务上表现比Adam差
为了直接建立"块异质性"和"为什么SGD比Adam差"之间的联系，作者考虑了一个人为构造的例子和一个真实世界的例子：

![](https://cdn.nlark.com/yuque/0/2024/png/406504/1728717153161-42ecf505-3898-445a-a72a-4437cbea3f56.png)

1. 人为构造的MLP：作者考虑了在MNIST上训练的4层MLP，通过缩放每一层来改变异质性程度。图5(a)显示，随着异质性的增加，SGD的表现逐渐变差。
2. MLP-mixer：这是一个著名的全MLP架构，在某些视觉任务上优于CNNs和ViTs。图5(b)(c)显示，MLP-mixer的初始Hessian具有块异质性，SGD在这种架构上落后于Adam。

### 2.5 预训练Transformer中块异质性的减少
作者指出，不同的Transformers表现出不同程度的块异质性。虽然所有检查的Transformers都显示出强烈的块异质性，但这种异质性可以得到缓解，从而减少SGD的性能下降。

![](https://cdn.nlark.com/yuque/0/2024/png/406504/1728717200133-dad95497-c7af-4776-ab5a-290ca7acd545.png)

如图6所示，在SFT任务上预训练的GPT2比从头开始预训练的GPT2（图4(f)）表现出更少的块异质性。在这种情况下，虽然SGD仍然比Adam慢，但最终达到了类似的损失。与从头开始训练GPT2相比（附录B中的图10(d)），SGD和Adam之间的性能差距显著缩小。

这些发现表明，由架构设计引起的异质性可以通过选择"好的"权重来缓解。这部分解释了为什么像SGD甚至其零阶版本这样的简单方法仍然可以有效地微调语言模型，尽管收敛速度较慢。

### 2.6 选择SGD或Adam的启示
作者提出了一个定量指标，可以在训练开始前预测SGD的不适用性。这个指标是初始化时块wise Hessian谱之间的平均JS距离，记为 $ JS_0 $。表1列出了各种模型的 $ JS_0 $ 值。$ JS_0 $ 建立了Transformer和CNNs损失景观之间的定量差异。此外，$ JS_0 $ 独立于优化器，可以在训练之前检查。

![](https://cdn.nlark.com/yuque/0/2024/png/406504/1728717239776-677d6165-d527-407f-b590-0d28f4061987.png)

为了验证 $ JS_0 $ 的有效性，作者总结了不同模型的 $ JS_0 $ 和相应的SGD性能，如图7所示。他们发现，随着 $ JS_0 $ 的增加，SGD和Adam之间的性能差距变大。因此，$ JS_0 $ 可以作为预测SGD是否可能表现不如Adam的潜在指标。

![](https://cdn.nlark.com/yuque/0/2024/png/406504/1728717268682-5299d13e-4261-4526-b9be-ece9eaa7e0da.png)

## 3. 二次模型的案例研究和初步理论
### 3.1 实验观察
作者研究了具有块对角Hessian的二次函数，有和没有块异质性两种情况。他们考虑以下二次最小化问题：

$ \min_{w \in \mathbb{R}^d} L(w) = \frac{1}{2}w^T H w - h^T w $

其中 $ H \in \mathbb{R}^{d \times d} $ 是正定矩阵，$ h \in \mathbb{R}^d $。$ H $ 是一个块对角矩阵：$ H = \text{diag}(H_1, \cdots, H_L) $，其中 $ H_l \in \mathbb{R}^{d_l \times d_l} $ 且 $ d = \sum_{l=1}^L d_l $。作者考虑了四种类型的Hessian $ H $，所有情况下条件数都设置为5000：

1. Transformer类型谱的Hessian
2. CNN类型谱的Hessian
3. 简化的异质谱Hessian
4. 简化的同质谱Hessian

他们研究了两类优化器：

1. 单一学习率优化器：梯度下降（GD）

$ w^{t+1} = w^t - \eta\nabla L(w) = w^t - \eta(Hw^t - h) $

2. 坐标wise学习率优化器：Adam（简化版本，无偏差修正）

$ w^{t+1} = w^t - \eta(D^0_{Adam})^{-1}\nabla L(w) = w^t - \eta(D^0_{Adam})^{-1}(Hw^t - h) $

其中 $ D^0_{Adam} = \text{diag}(\nabla L(w^0) \circ \nabla L(w^0))^{\frac{1}{2}} $ 且 $ \nabla L(w^0) = Hw^0 - h $。

实验观察总结：

+ 对于具有异质块的Hessian（情况1和3），GD明显落后于Adam。
+ 对于具有同质块的Hessian（情况2和4），GD与Adam表现相当。

作者假设GD表现不佳是因为它对所有块使用单一学习率，无法处理块之间的异质性。这种异质性可以通过使用不同的学习率来更好地处理，这正是Adam的设计特点。

### 3.2 初步理论结果
作者提供了初步的理论结果，描述了GD如何在具有异质Hessian的问题上落后于Adam。他们首先给出了GD的下界：

**命题1（GD的下界）**

考虑最小化 $ L(w) = \frac{1}{2}w^T H w - h^T w $，其中 $ H \in \mathbb{R}^{d \times d} $ 是正定矩阵，$ h \in \mathbb{R}^d $。令 $ w_{GD}^t $ 为GD $ t $ 步后的输出。存在一个块对角矩阵 $ H $，$ h $ 和一个初始点 $ w^0 $，使得对任意 $ \eta $，都有：

$ L(w_{GD}^{t+1}) - L^* \geq \left(1 - \frac{2}{\kappa + 1}\right)(L(w_{GD}^t) - L^*) $

其中 $ \kappa $ 是 $ H $ 的条件数。命题1表明GD的复杂度为 $ \tilde{O}(\kappa) $，且这个复杂度是紧的。

接下来，作者证明Adam可以实现更好的复杂度。这是因为Adam通过其对角预处理器 $ D_{Adam}^0 $ 为不同的块子矩阵 $ H_l $ 选择不同的学习率。作者考虑了覆盖常用分布（如高斯分布、均匀分布等）的通用随机初始化。

假设1（随机初始化）假设初始化 $ w^0 $ 是从一个连续分布中采样的，即（由 $ w^0 $ 诱导的）任何零Lebesgue测度集的概率为0。

**定理1（Adam with **$ \beta_2 = 1 $** 的上界）**

考虑与命题1相同的设置，考虑 $ \beta_1 = 0 $ 且 $ \beta_2 = 1 $ 的Adam，如式(2)所示。假设初始化满足假设1。令 $ w_{Adam}^t $ 为Adam $ t $ 步后的输出。令 $ \eta = \min_{l \in [L]} \frac{1}{C_{l,1}} $。那么以概率1，我们有：

$ L(w_{Adam}^{t+1}) - L^* \leq \max_{l \in [L]}\left(1 - \frac{1}{\kappa_{Adam,l}}\right)(L(w_{Adam}^t) - L^*) $

其中 $ \kappa_{Adam,l} = r\kappa_l $，$ \kappa_l $ 是 $ H_l $ 的条件数，常数 $ r $ 与 $ w^0 $ 相关，定义为：

$ r = \frac{\max_{l \in [L]} C_{l,2}^2}{\min_{l \in [L]} C_{l,1}^2}, \text{其中} C_{l,1} = \min_{i \in [d_l]}\frac{|[\nabla L(w^0)]_{l,i}|}{\lambda_{l,1}}, C_{l,2} = \max_{i \in [d_l]}\frac{|[\nabla L(w^0)]_{l,i}|}{\lambda_{l,1}} $

定理1表明Adam（with $ \beta_2 = 1 $）的复杂度为 $ \tilde{O}(r \cdot \max_{l \in [L]} \kappa_l) $。系数 $ r $ 取决于每个块的初始梯度与主特征值之比，较小的比值会带来更快的收敛速度。

作者进一步指出，$ \beta_2 = 1 $ 的条件是必要的，因为任何 $ \beta_2 < 1 $ 都会导致非收敛问题。他们重述了先前研究的结果：

**命题2（常数学习率Adam with **$ \beta_2 < 1 $** 的非收敛性）**

考虑最小化 $ L(w) = \frac{1}{2}w^2, w \in \mathbb{R} $。考虑 $ \beta_1 = 0 $ 且 $ \beta_2 < 1 $ 的Adam，如式(3)所示。令 $ w_{Adam}^t $ 为Adam $ t $ 步后的输出。对于式(3)存在一个离散极限环，且

$ \lim\inf_{t \to \infty} (L(w_{Adam}^t) - L^*) > 0 $

现在，作者比较了Adam和GD的复杂度。根据定理1，当 $ r \cdot \max_{l \in [L]} \kappa_l \leq \kappa $ 时，Adam比GD更快。在具有异质块的二次模型（情况3）中，作者的模拟表明，使用标准高斯随机初始化时，以 $ \frac{2}{3} $ 的概率 $ r \leq 1000 $。由于 $ \max_{l \in [L]} \kappa_l \approx 1 $，我们有 $ r \cdot \max_{l \in [L]} \kappa_l \leq 1000 $，大概率下比 $ \kappa = 5000 $ 小约5倍。因此，Adam可能比GD快5倍左右，这在图8中得到了验证，Adam明显优于GD。

![](https://cdn.nlark.com/yuque/0/2024/png/406504/1728717409731-9b5c0e83-8ece-4ce4-9506-f27d5d8c04f6.png)

作者在表2中总结了GD和Adam的复杂度：

| 优化器 | GD | Adam with $ \beta_1 = 0 $ and $ \beta_2 = 1 $ (2) | Adam with $ \beta_1 = 0 $ and $ \beta_2 < 1 $ (3) |
| --- | --- | --- | --- |
| 复杂度 | $ \tilde{O}(\kappa) $ | $ \tilde{O}(r \cdot \max_{l \in [L]} \kappa_l) $ | ✗ (不收敛) |


其中 $ \kappa $ 和 $ \kappa_l $ 分别表示全Hessian和块子矩阵的条件数，$ r $ 如前面定义。

作者指出，虽然还有改进Adam复杂度上界的空间，但这可能具有挑战性。他们提供了一些技术讨论，指出如果没有 $ H_l $ 的额外结构，改进因子 $ r $ 可能会很困难。关键的技术步骤是限制预处理矩阵的条件数 $ \kappa((D_{Adam,l}^0)^{-1}H_l) $。直观上，当 $ H_l $ 本身具有近似对角结构（如纯对角、三对角或对角占优）时，对角预处理器对 $ H_l $ 最有效。然而，目前还不清楚这些结构是否在Transformer中成立。

尽管Adam的预处理可能不总是减少"局部"条件数 $ \kappa_l $，但复杂度中的系数现在独立于"全局"条件数 $ \kappa $。如前所述，这种系数的变化可能导致对GD的显著改进。这种复杂度的改进归因于Hessian的块对角结构及其异质的块wise谱。

作者的理论表明：对于具有块异质性的问题，像GD这样的单一学习率方法可能会大大落后于像Adam这样的坐标wise学习率方法。

## 4. 结论
这篇论文探讨了为什么SGD在Transformer上的表现明显不如Adam。主要发现如下：

1. 通过引入"块异质性"的概念，作者解释了SGD在Transformer上表现不佳的原因。块异质性指的是不同参数块之间的Hessian谱的巨大差异。
2. 作者在各种Transformers、CNNs和MLPs上验证了块异质性阻碍SGD性能的现象。在具有块异质性的问题上，SGD始终表现比Adam差；而在没有块异质性的问题上，SGD可以与Adam表现相当。
3. 在二次模型上的理论分析显示，GD在具有块异质性的问题上可能比Adam慢。这是因为GD对所有块使用单一学习率，而Adam通过为不同块分配不同学习率来缓解这个问题。
4. 作者提出了一个定量指标 $ JS_0 $（初始化时块wise Hessian谱之间的平均JS距离），可以在训练开始前预测SGD的不适用性。

这项研究为理解Transformer训练和更广泛的神经网络优化提供了新的视角。通过引入块wise Hessian谱的概念，作者揭示了Transformer和CNN之间的本质差异，这可能对未来的模型设计和优化策略产生影响。

然而，这项研究也存在一些局限性。例如，由于计算资源的限制，作者无法在更大规模的模型上验证他们的发现。此外，虽然他们提供了初步的理论分析，但还需要更深入的理论工作来完全解释观察到的现象。

未来的研究方向可能包括：

1. 在更大规模的模型上验证块异质性现象。
2. 开发新的优化算法，能更好地处理块异质性。
3. 探索块异质性与其他Transformer现象（如梯度消失/爆炸）之间的潜在联系。
4. 研究如何通过架构设计或初始化策略来减少块异质性，从而改善SGD在Transformer上的表现。



