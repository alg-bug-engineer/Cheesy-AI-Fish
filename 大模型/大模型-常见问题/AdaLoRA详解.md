### <font style="color:rgb(18, 18, 18);">一、背景</font>
<font style="color:rgb(0, 0, 0);">论文标题：ADAPTIVE BUDGET ALLOCATION FOR PARAMETER- EFFICIENT FINE-TUNING</font>

<font style="color:rgb(0, 0, 0);">论文链接：https://arxiv.org/pdf/2303.10512.pdf</font>

<font style="color:rgb(0, 0, 0);">参考链接: https://zhuanlan.zhihu.com/p/657130029</font>

#### <font style="color:rgb(0, 0, 0);">LORA缺陷</font>
<font style="color:rgb(83, 88, 97);">LORA是一种有效的低资源方式，仅仅</font><font style="color:rgba(252,7,20,1);">微调万分之一参数就能达到全量参数微调的效果</font><font style="color:rgb(83, 88, 97);">。但本文指出</font>**<font style="color:rgb(83, 88, 97);">LORA是</font>****<font style="color:#601BDE;">将可微调参数平均分布在每个权重矩阵上</font>****<font style="color:rgb(83, 88, 97);">，即</font>**<font style="color:#601BDE;">预先规定了每个增量矩阵</font><font style="color:#601BDE;">Δ</font><font style="color:#601BDE;">的秩</font><font style="color:#601BDE;">r</font><font style="color:#601BDE;">必须相同。</font>**<font style="color:rgb(83, 88, 97);">忽略了</font>**<font style="color:#601BDE;">不同层、不同类型参数对下游任务的重要程度</font>**<font style="color:rgb(83, 88, 97);">，因此LORA微调的效果可能不是最优的</font>**<font style="color:rgb(83, 88, 97);">。</font>

#### <font style="color:rgb(0, 0, 0);">AdaLORA的提出</font>
<font style="color:rgb(18, 18, 18);">AdaLoRA </font><font style="color:rgb(0, 0, 0);">改进了LORA可微调参数的分配方式，</font><font style="color:#601BDE;">根据每个参数的重要程度自动得为其分配可微调参数的预算</font><font style="color:rgb(0, 0, 0);">。</font><font style="color:#DF2A3F;">LoRA中是让模型学习BA，去近似SVD分解的结果</font><font style="color:rgb(0, 0, 0);">，但是在训练过程中，</font><font style="color:rgba(252,7,20,1);">没有引入任何</font>**<font style="color:rgba(252,7,20,1);">SVD分解相关的性质</font>**<font style="color:rgba(252,7,20,1);">做约束</font><font style="color:rgb(0, 0, 0);">，而AdaLoRA则是直接将这一束缚考虑到了Loss中。</font>

<font style="color:rgb(0, 0, 0);">具体地：</font>

    - <font style="color:rgb(25, 27, 31);">基于</font><font style="color:rgba(252,7,20,1);">奇异值分解(SVD)的形式参数化增量更新</font><font style="color:rgb(25, 27, 31);">，将增量矩阵以奇异值分解的形式表达，规避了大量SVD运算；</font>
    - <font style="color:rgb(25, 27, 31);">基于设计的</font>**<font style="color:rgba(252,7,20,1);">重要程度的参数分配（</font>**<font style="color:rgba(252,7,20,1);">importance-aware rank allocation</font>**<font style="color:rgba(252,7,20,1);">）</font>**<font style="color:rgb(25, 27, 31);">方法，来高效裁剪</font><font style="color:rgb(0, 0, 0);">不重要</font><font style="color:rgb(25, 27, 31);">奇异值，从而减少计算开销</font>

**<font style="color:rgb(0, 0, 0);">核心Idea</font>**<font style="color:rgb(0, 0, 0);">：AdaLoRA adjusts the </font>**<font style="color:rgb(0, 0, 0);">rank</font>**<font style="color:rgb(0, 0, 0);"> of incremental matrices to control their budget.</font>

### 二、<font style="color:rgb(0, 0, 0);">AdaLORA</font>原理
#### 研究方法
![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707099701938-6c6fef21-008d-44da-bd96-15cac9acec84.png)

<font style="color:rgb(25, 27, 31);"> </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707202059732-7d1b7dea-861d-4c8b-9596-b06fc325720b.png)**<font style="color:rgb(25, 27, 31);">表示预测结果和真实标签间的差异，P和Q都必须是正交矩阵，即</font>**![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707202156929-74f206c7-736c-4d5b-ab54-3185c720ef07.png)

#### <font style="color:rgb(0, 0, 0);">方法涉及三个具体问题:</font>
+ <font style="color:black;">AdaLoRA 中</font>**<font style="color:black;">Loss</font>**<font style="color:black;">的设计（上面讲了）</font>
+ <font style="color:black;">AdaLoRA 中</font>**<font style="color:black;">重要性分数计算</font>**<font style="color:black;">的设计</font>
+ <font style="color:black;">AdaLoRA 中</font>**<font style="color:black;">如何根据重要性分数筛选不重要的三元组，并调节矩阵的秩</font>**

##### <font style="color:rgb(25, 27, 31);">如何建模特征的重要性？</font>
<font style="color:rgb(0, 0, 0);">AdaLoRA的整体目标是</font><font style="color:rgba(252,7,20,1);">要做</font>**<font style="color:rgba(252,7,20,1);">参数预算（budget）</font>**<font style="color:rgba(252,7,20,1);">，即忽略不重要的参数，把训练资源给重要的参数</font><font style="color:rgb(0, 0, 0);">，在AdaLoRA中，通过</font><font style="color:rgba(252,7,20,1);">“</font>**<font style="color:rgba(252,7,20,1);">变秩</font>**<font style="color:rgba(252,7,20,1);">”</font><font style="color:rgb(0, 0, 0);">来实现这种预算的动态分配的。</font>

tips：<font style="color:rgb(0, 0, 0);">为什么不直接修改Lora的r值-</font>**<font style="color:#601BDE;">因为r代表的是超参数无法动态调整</font>**<font style="color:rgb(0, 0, 0);">。</font>

**<font style="color:#601BDE;">（一）单参数重要性分数设计：</font>**

<font style="color:rgb(0, 0, 0);">AdaLoRA作者就提出了这样一种计算t时刻, 单个模型参数（</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707102156780-74d8cf83-432e-43e2-a52b-fe9839e222e6.png)<font style="color:rgb(0, 0, 0);">）重要性的方法。</font>

<font style="color:rgb(0, 0, 0);">首先给出公式，即用</font><font style="color:rgb(25, 27, 31);">敏感性</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707204565777-39d05b8a-6d5e-4124-8d77-1d05ec0ea90e.png)<font style="color:rgb(25, 27, 31);">和不确定性 </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707204579836-e585473b-0cf5-4aa6-98e8-2e5e6a616aac.png)<font style="color:rgb(25, 27, 31);"> 的乘积来表示这个特征的重要性：</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707204591950-dc9c7a61-3423-4002-9af1-2081b83aba65.png)

**<font style="color:#2F4BDA;">1、敏感性</font>**

    1. **<font style="color:rgb(0, 0, 0);">一个最直观的想法就是：去看看这个参数对Loss的影响</font>**<font style="color:rgb(0, 0, 0);">。</font><font style="color:rgb(25, 27, 31);">在模型剪枝中，单个参数的</font>**<font style="color:rgba(252,7,20,1);">敏感性</font>**<font style="color:rgb(25, 27, 31);">被定义为</font><font style="color:#601BDE;">梯度和权重乘积</font><font style="color:rgb(25, 27, 31);">的绝对值，如下式：</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707205140015-57e4da02-d6b8-4764-8b5a-fb25654ed75b.png)<font style="color:rgb(25, 27, 31);">，其中</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707204222901-fd51a64c-4a96-4f19-80b4-bfbee1a3bd5b.png)<font style="color:rgb(25, 27, 31);"> 是任意可训练的权重，</font>

<font style="color:rgb(25, 27, 31);"> </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707204211848-3ce75ff5-0c50-44d3-9474-d8aeefe49dfd.png)<font style="color:rgb(25, 27, 31);"> 是这个权重的梯度。</font>

    2. <font style="color:rgb(25, 27, 31);">在SGD中，这个重要性只是</font><font style="color:rgb(0, 0, 0);">mini-batch</font><font style="color:rgb(25, 27, 31);">的样本反应的重要性，</font><font style="color:#601BDE;">不同step间重要性分数</font><font style="color:rgb(0, 0, 0);">可能会受到mini-batch客观波动的影响，</font><font style="color:rgb(25, 27, 31);">我们可以使用</font><font style="color:#601BDE;">滑动平均思想（代表性：</font>**<font style="color:#601BDE;">momentum</font>**<font style="color:rgb(25, 27, 31);">）来减轻</font><font style="color:rgb(0, 0, 0);">mini-batch</font><font style="color:rgb(25, 27, 31);">带来的重要性的评估误差，表示为式：</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707204314933-0fcfaf71-7fa2-47ad-81ce-9c409fbae6bf.png)

<font style="color:rgb(25, 27, 31);">其中 </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707204371668-ee049e4c-249c-420f-8716-df2213cc59ab.png)<font style="color:rgb(25, 27, 31);"> 代表的是</font><font style="color:#601BDE;">训练步</font><font style="color:rgb(25, 27, 31);">数， </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707204381857-6e02d8a4-eda3-4922-8206-26b7760755a0.png)<font style="color:rgb(25, 27, 31);">是</font><font style="color:#601BDE;">滑动平均中控制历史记录和当前批次占比的超参</font><font style="color:rgb(25, 27, 31);">数。</font>

**<font style="color:#2F4BDA;">2、不确定性</font>**

    1. <font style="color:rgb(25, 27, 31);">有了重要性，我们可以计算敏感性的</font>**<font style="color:#601BDE;">不确定性</font>**<font style="color:rgb(25, 27, 31);">（Uncertainty）</font>[[4]](https://zhuanlan.zhihu.com/p/667432109#ref_4)<font style="color:rgb(25, 27, 31);">，不确定性是AdaLoRA的作者在他的另外一个论文Platon[4]中提出的指标，它表示的是</font><font style="color:#601BDE;">敏感性的局部时间变化</font><font style="color:rgb(25, 27, 31);">，定义为 ：</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707204412714-1629cb37-2a48-41cf-8c3e-e8a703afb631.png)

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707271612609-1b8b972e-1c38-433f-90fa-51baef141053.png)

<font style="color:rgb(25, 27, 31);">对于不确定性，我们最好也对它进行滑动平局：</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707204512371-a8013762-555f-40c0-96f1-62164e2b8a6a.png)

**<font style="color:#2F4BDA;">3、单参数重要性</font>**

    1. <font style="color:rgb(25, 27, 31);">最终，我们可以使用敏感性</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707204565777-39d05b8a-6d5e-4124-8d77-1d05ec0ea90e.png)<font style="color:rgb(25, 27, 31);">和不确定性 </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707204579836-e585473b-0cf5-4aa6-98e8-2e5e6a616aac.png)<font style="color:rgb(25, 27, 31);"> 的乘积来表示这个特征的重要性：</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707204591950-dc9c7a61-3423-4002-9af1-2081b83aba65.png)

**<font style="color:#601BDE;">（二）三元组重要性分数</font>**<font style="color:#601BDE;">：</font>

+ <font style="color:rgb(25, 27, 31);">对于三元组 </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707204620122-c03b73b1-41dc-4c7a-bcc8-2f38e83faa13.png)<font style="color:rgb(25, 27, 31);"> ，它的重要性是三元组三个值的加权和，权值取决于 </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707204634888-abe1111e-a47d-4f04-8797-6ded4930582c.png)<font style="color:rgb(25, 27, 31);"> </font>

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707204659899-e8157f92-8a02-4cc3-aed3-c53c7c92d247.png)

<font style="color:rgb(0, 0, 0);"> </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707099223539-d3fde841-c251-4c0b-bb2b-7944633facee.png)<font style="color:rgb(0, 0, 0);">分别表示“第i列”和“第i行”。</font>

**<font style="color:rgb(0, 0, 0);">即：三元组的重要性分数 =</font>**<font style="color:rgb(0, 0, 0);">  </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707100337933-4d98d200-dc5c-486e-9485-2a901c8e3086.png)<font style="color:rgb(0, 0, 0);">的重要性分数 + P矩阵列中所有元素重要性分数的均值 + Q矩阵行中所有元素重要性分数的均值。</font><font style="color:#601BDE;">取均值的原因，是不希望参数量影响到重要性分数</font><font style="color:rgb(0, 0, 0);">。</font>

##### <font style="color:rgb(25, 27, 31);">如何根据重要性自动计算秩 </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707205502867-9dfb60c5-2a61-45a7-8744-a7d6a875ec28.png)<font style="color:rgb(25, 27, 31);"> 的值，进行</font>**<font style="color:rgb(25, 27, 31);">动态调整矩阵秩</font>**<font style="color:rgb(25, 27, 31);">？</font>
<font style="color:rgb(25, 27, 31);">为了根据重要性计算秩的值，</font><font style="color:rgba(252,7,20,1);">一个最直观的方式是将 </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707205502867-9dfb60c5-2a61-45a7-8744-a7d6a875ec28.png)<font style="color:rgba(252,7,20,1);"> 看做模型的一个参数，然后根据模型的损失值来调整 </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707205502867-9dfb60c5-2a61-45a7-8744-a7d6a875ec28.png)<font style="color:rgba(252,7,20,1);"></font>**<font style="color:rgb(25, 27, 31);">核心是：根据三元组重要性分数，对</font>**<font style="color:rgb(25, 27, 31);"> </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707203582106-eb9487d7-873c-4727-8248-b9f48ea00161.png)<font style="color:rgb(25, 27, 31);"> </font>**<font style="color:rgb(25, 27, 31);">矩阵中相应的</font>**![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707203592454-c11a12f1-1447-495d-afb4-60b6728a5857.png)<font style="color:rgb(25, 27, 31);"> </font>**<font style="color:rgb(25, 27, 31);">做置0处理。</font>**

**<font style="color:#601BDE;">给出符号定义:</font>**

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707100367118-0b1cf03d-ab52-45db-9c10-287f84757b4b.png)

**<font style="color:rgb(25, 27, 31);"></font>**

**<font style="color:#601BDE;">步骤如下：</font>**

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707203614924-735bd2b9-6a55-4552-9f05-5f2140583726.png)

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707203893560-f6274324-c287-4a53-a869-bd11168584e6.png)

##### <font style="color:rgb(0, 0, 0);">top-b策略</font>
**<font style="color:#601BDE;">策略总结：</font>**

**<font style="color:rgb(0, 0, 0);">过程就和</font>****<u><font style="color:rgb(0, 0, 0);">warm-up</font></u>****<font style="color:rgb(0, 0, 0);">非常相似，在训练刚开始，我们逐渐增加top_b，也就是逐渐加秩，让模型尽可能多探索。到后期再慢慢把top_b降下来，直到最后以稳定的top_b进行训练，达到AdaLoRA的总目的：把训练资源留给最重要的参数。</font>**

**<font style="color:#601BDE;">具体细节：</font>**

<font style="color:rgb(25, 27, 31);">具体来讲，</font><font style="color:rgba(252,7,20,1);">在最开始的 </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707205776300-7505c99e-0e1e-44a0-9459-ffc928a3cae9.png)<font style="color:rgba(252,7,20,1);"> 步，我们给预算一个稍微大的值</font><font style="color:rgb(25, 27, 31);">，让模型快速达到比较好的效果。接下来的 </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707205789792-869f0e26-e51d-4e2e-975c-dfe0b5a2cbd9.png)<font style="color:rgb(25, 27, 31);"> 步，我们通过让</font><font style="color:rgba(252,7,20,1);">秩的预算以三次方的速度逐渐减小，来达到对秩进行剪枝的目的</font><font style="color:rgb(25, 27, 31);">。在最后剩下的 </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707205807299-e5493216-a467-4933-8a2b-f264a4fd9b47.png)<font style="color:rgb(25, 27, 31);">步中，我们稳</font><font style="color:rgba(252,7,20,1);">定秩的大小来让模型效果达到当前秩上的一个局部最优解</font><font style="color:rgb(25, 27, 31);">。</font>

<font style="color:rgb(25, 27, 31);"></font>

<font style="color:rgb(25, 27, 31);">预算的完整计算方式如式：</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707203455896-2cfbbfbe-ad69-4505-a1be-0232d7b9e692.png)

#### <font style="color:rgb(0, 0, 0);">具体流程</font>
**<font style="color:rgb(25, 27, 31);">AdaLoRA变秩的整体流程如下：</font>**

1. **<font style="color:rgb(0, 0, 0);">首先，初始化三个矩阵</font>**![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707099079484-c3689886-0e27-49ee-ae34-40c681a5a910.png)<font style="color:rgb(0, 0, 0);">  。其中</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707099110291-16daa77d-7629-41c7-83b7-2edaf58aeaae.png)<font style="color:rgb(0, 0, 0);">矩阵比较特殊，其大部分元素为0，只有对角线上的r个元素有值，</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707099122309-346aac5b-91e9-4810-9e16-c6469a9578ca.png)<font style="color:rgb(0, 0, 0);"> 。初始化时，</font>**<font style="color:rgb(0, 0, 0);">将</font>**![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707099110291-16daa77d-7629-41c7-83b7-2edaf58aeaae.png)<font style="color:rgb(0, 0, 0);"> </font>**<font style="color:rgb(0, 0, 0);">初始化为0，</font>**<font style="color:rgb(0, 0, 0);">  P和Q </font>**<font style="color:rgb(0, 0, 0);">初始化为高斯随机矩阵</font>**<font style="color:rgb(0, 0, 0);">，这样做的目的为了</font><font style="color:#601BDE;">在训练开始保证</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707099159461-c8acfe7d-a138-4e3e-b6cb-59590e482389.png)<font style="color:#601BDE;">是0，以此避免引入噪声。</font>
2. <font style="color:rgb(0, 0, 0);">然后，</font>**<font style="color:rgb(0, 0, 0);">正常做forward和backward，得到Loss和参数的梯度</font>**<font style="color:rgb(0, 0, 0);">。</font>
3. <font style="color:rgb(0, 0, 0);">接着，</font>**<font style="color:rgb(0, 0, 0);">根据Loss和参数梯度</font>**<font style="color:rgb(0, 0, 0);">，对每个三元组(triplets) </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707099185546-3b8918b2-c769-4567-b9bb-509e6e377d7a.png)<font style="color:rgb(0, 0, 0);"> </font>**<font style="color:rgb(0, 0, 0);">计算重要性分数</font>**<font style="color:rgb(0, 0, 0);">。 </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707099223539-d3fde841-c251-4c0b-bb2b-7944633facee.png)<font style="color:rgb(0, 0, 0);">分别表示“第i列”和“第i行”。</font>
4. <font style="color:rgb(0, 0, 0);">接着，</font>**<font style="color:rgb(0, 0, 0);">根据计算出来的重要性分数，将不重要的三元组挑选出来</font>**<font style="color:rgb(0, 0, 0);">。</font>
5. <font style="color:rgb(0, 0, 0);">接着，</font>**<font style="color:rgb(0, 0, 0);">对于不重要的三元组，将</font>**![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707099260749-0aef1d76-439a-4a34-94f8-4be863c11409.png)**<font style="color:rgb(0, 0, 0);">其值置0</font>**<font style="color:rgb(0, 0, 0);">。这样，在下一次做forward时，这个三元组里对应的P向量和Q向量</font>**<font style="color:rgb(0, 0, 0);">相当于</font>**<font style="color:rgb(0, 0, 0);">被mask掉了，对Loss没有贡献。也就起到了</font>**<font style="color:rgb(0, 0, 0);">变秩</font>**<font style="color:rgb(0, 0, 0);">的效果。</font>
    1. **<font style="color:rgb(25, 27, 31);"></font>****<font style="color:rgba(252,7,20,1);">为什么只是将</font>**<font style="color:rgba(252,7,20,1);"> </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707099260749-0aef1d76-439a-4a34-94f8-4be863c11409.png)<font style="color:rgba(252,7,20,1);"> </font>**<font style="color:rgba(252,7,20,1);">置0，而不是把整个三元组删掉？</font>**<font style="color:rgb(25, 27, 31);">模型的学习是一个探索的过程，</font>**<font style="color:rgb(25, 27, 31);">在一开始模型认为不重要的三元组，在后续过程中模型可能会慢慢学到它的重要性。因此，mask是比删掉更合理的方法</font>**<font style="color:rgb(25, 27, 31);">。也正是这个原因，我们在步骤6中，不管三元组有没有被mask掉，我们都会正常用梯度更新P和Q。</font>
6. <font style="color:rgb(0, 0, 0);">接着，使用  2 中计算出来的梯度，更新P和Q的参数。</font>
7. <font style="color:rgb(0, 0, 0);">然后，使用更新完毕的  ，开启新一轮forward和backward，重复上面步骤，随时动态更新参数的秩。</font>

![](https://cdn.nlark.com/yuque/0/2024/jpeg/35381469/1707098877611-6f9c49ba-ab84-4ca9-9265-5e4be0f66458.jpeg)

#### <font style="color:rgb(0, 0, 0);">与LoRA对比</font>
<font style="color:rgb(0, 0, 0);">相比LORA，AdaLORA这种设计方式有两个优点：</font>

+ <font style="color:rgb(1, 1, 1);">AdaLORA只裁剪奇异值矩阵，并不裁剪奇异向量，因此训练过程中</font><font style="color:#601BDE;">更容易恢复被误删的奇异值。</font>
+ <font style="color:rgb(1, 1, 1);">AdaLORA的P和Q正交矩阵，而LORA的A和B非正交。AdaLORA训练过程中裁剪操作不会影响其他奇异值对应的奇异向量，因此</font><font style="color:#601BDE;">训练会更稳定泛化性能更好</font><font style="color:rgb(1, 1, 1);">。</font>**文档内容概述：AdaLoRA详解**





