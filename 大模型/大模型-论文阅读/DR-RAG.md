# 一、RAG研究背景与DR-RAG动机

近年来,大型语言模型(Large Language Models, LLMs)在自然语言处理(NLP)领域取得了显著进展,尤其是在文本生成任务上展现出惊人的能力。然而,当面对知识密集型任务如开放域问答(Open-domain Question Answering)时,LLMs仍然面临两大挑战:

1. 知识的准确性:由于LLMs主要依赖自身参数来存储知识,当面对超出其训练数据的问题时,容易产生与事实不符的"幻觉"(hallucination)现象。
2. 知识的时效性:LLMs的知识来自于固定的训练数据,难以应对实时更新的知识需求。

为了克服上述挑战,研究者提出了检索增强生成(Retrieval-Augmented Generation, RAG)框架。其核心思想是通过检索外部知识库,来补充和扩展LLMs的知识,从而提高生成内容的准确性和时效性。然而,现有的RAG方法在多跳问答(Multi-hop Question Answering)等复杂任务中仍然存在不足:

1. 低效:为了回答一个复杂问题,可能需要多次调用LLMs,导致计算成本高昂。
2. 不完备:通过单一检索很难找到回答问题所需的所有相关文档。

针对这些问题,本文提出了一种创新的RAG框架——动态相关检索增强生成(Dynamic-Relevant Retrieval-Augmented Generation, DR-RAG),旨在通过挖掘文档间的动态相关性,来提高检索召回率和答案准确率,同时兼顾整体效率。

# 二、DR-RAG技术方案

### 2.1 总体框架

DR-RAG的核心是一个两阶段检索框架,配合文档相关性分类器,来实现动态相关文档的高效筛选。

![https://files.mdnice.com/user/18983/49ec4ee3-3dc8-4d99-ae6c-42ac5cc09abb.png](https://files.mdnice.com/user/18983/49ec4ee3-3dc8-4d99-ae6c-42ac5cc09abb.png)

![https://cdn.nlark.com/yuque/0/2024/png/406504/1718670848863-226cf677-2a10-43fa-a9ff-b79c438f03f6.png](https://cdn.nlark.com/yuque/0/2024/png/406504/1718670848863-226cf677-2a10-43fa-a9ff-b79c438f03f6.png)

如图1所示,DR-RAG主要包含三个步骤:

1. 首次检索(First-retrieval Stage):通过查询与文档的相似度匹配,检索出k1个静态相关文档(Static-Relevant Document)。
2. 再次检索(Second-retrieval Stage):将查询和每个静态相关文档拼接,生成新的查询,用于检索动态相关文档(Dynamic-Relevant Document)。
3. 分类筛选(Selection Process):通过相关性分类器评估每个动态文档的价值,并按照前向或反向策略进行筛选,最终输入LLMs生成答案。

### 2.2 查询-文档拼接

DR-RAG的一个关键创新点是查询-文档拼接(Query Document Concatenation, QDC)。传统的检索方法很难发现隐含的动态相关文档,如图2的例子:

![https://cdn.nlark.com/yuque/0/2024/png/406504/1718670865012-47b66092-b115-4ee1-916d-3f5b3509d5d4.png](https://cdn.nlark.com/yuque/0/2024/png/406504/1718670865012-47b66092-b115-4ee1-916d-3f5b3509d5d4.png)

可以看到,动态相关文档和原始查询的直接关联度很低,因为知识库中关于"配偶"的内容太多。但如果把从静态文档中提取出的关键信息"Johan Ludvig Heiberg"再拼接到查询中,就能准确地检索出与之最相关的动态文档了。

QDC的具体过程可以用以下公式表示:

[](https://www.yuque.com/api/services/graph/generate_redirect/latex?%5Cbegin%7Baligned%7D%0Ad_m%20%26%3D%20Retriever(q)%2C%20m%20%5Cin%20%5B1%2C%20k_1%5D%20%5C%5C%0Aq_m%5E*%20%26%3D%20Concat(q%2C%20d_m)%20%5C%5C%20%0Ad_n%20%26%3D%20Retriever(q_m%5E*)%2C%20n%20%5Cin%20%5Bk_1%2B1%2C%20k_1%2Bk_2%5D%20%5C%5C%0Aanswer%20%26%3D%20LLM(Concat(d_1%2C%20d_2%2C%20...%2C%20d_%7Bk1%2Bk2%7D%2C%20q))[https://www.yuque.com/api/services/graph/generate_redirect/latex?\begin{aligned} d_m %26%3D Retriever(q)%2C m \in [1%2C k_1] \\ q_m^* %26%3D Concat(q%2C d_m) \\ d_n %26%3D Retriever(q_m^*)%2C n \in [k_1%2B1%2C k_1%2Bk_2] \\ answer %26%3D LLM(Concat(d_1%2C d_2%2C ...%2C d_{k1%2Bk2}%2C q)](https://www.yuque.com/api/services/graph/generate_redirect/latex?%5Cbegin%7Baligned%7D%0Ad_m%20%26%3D%20Retriever(q)%2C%20m%20%5Cin%20%5B1%2C%20k_1%5D%20%5C%5C%0Aq_m%5E*%20%26%3D%20Concat(q%2C%20d_m)%20%5C%5C%20%0Ad_n%20%26%3D%20Retriever(q_m%5E*)%2C%20n%20%5Cin%20%5Bk_1%2B1%2C%20k_1%2Bk_2%5D%20%5C%5C%0Aanswer%20%26%3D%20LLM(Concat(d_1%2C%20d_2%2C%20...%2C%20d_%7Bk1%2Bk2%7D%2C%20q)))%0A%5Cend%7Baligned%7D%0A%0A%5Cend%7Baligned%7D%0A)

其中,

[https://www.yuque.com/api/services/graph/generate_redirect/latex?Retriever](https://www.yuque.com/api/services/graph/generate_redirect/latex?Retriever)

表示基于相似度匹配的检索器,

[https://www.yuque.com/api/services/graph/generate_redirect/latex?LLM](https://www.yuque.com/api/services/graph/generate_redirect/latex?LLM)

表示用于生成答案的大语言模型。最终答案来自于拼接了所有相关文档的增强查询。

### 2.3 动态文档筛选

尽管QDC方法能提高检索召回率,但也可能引入一些冗余或无关文档。为了进一步优化,DR-RAG设计了一个轻量级的相关性分类器。给定一个查询

[https://www.yuque.com/api/services/graph/generate_redirect/latex?q](https://www.yuque.com/api/services/graph/generate_redirect/latex?q)

和两个候选文档

[https://www.yuque.com/api/services/graph/generate_redirect/latex?d_i%2Cd_j](https://www.yuque.com/api/services/graph/generate_redirect/latex?d_i%2Cd_j)

,分类器的目标是学习一个二分类模型:

[](https://www.yuque.com/api/services/graph/generate_redirect/latex?C(q%2Cd_i%2Cd_j)%20%3D%20%0A%5Cbegin%7Bcases%7D%0A1%20%26%20d_i%E5%92%8Cd_j%E9%83%BD%E4%B8%8Eq%E9%AB%98%E5%BA%A6%E7%9B%B8%E5%85%B3%20%5C%5C%20%0A0%20%26%20%E5%85%B6%E4%BB%96%E6%83%85%E5%86%B5%0A%5Cend%7Bcases%7D%0A)[https://www.yuque.com/api/services/graph/generate_redirect/latex?C(q%2Cd_i%2Cd_j) %3D \begin{cases} 1 %26 d_i和d_j都与q高度相关 \\ 0 %26 其他情况 \end{cases}](https://www.yuque.com/api/services/graph/generate_redirect/latex?C(q%2Cd_i%2Cd_j)%20%3D%20%0A%5Cbegin%7Bcases%7D%0A1%20%26%20d_i%E5%92%8Cd_j%E9%83%BD%E4%B8%8Eq%E9%AB%98%E5%BA%A6%E7%9B%B8%E5%85%B3%20%5C%5C%20%0A0%20%26%20%E5%85%B6%E4%BB%96%E6%83%85%E5%86%B5%0A%5Cend%7Bcases%7D%0A)

基于分类器,DR-RAG提出了两种动态文档筛选策略:

1. 反向筛选(Classifier Inverse Selection, CIS): 对第一阶段检索的所有文档,两两配对输入分类器,当某个文档与其他所有文档的分类结果都为0时,则该文档很可能是冗余的,需要删除。
2. 正向筛选(Classifier Forward Selection, CFS):在第二阶段检索中,对每个静态相关文档,逐一将其与动态文档配对输入分类器,只保留分类为1的动态文档。这相当于为每个静态文档设置了一个动态相关文档的"关注列表"。

# 三、实验评估

### 3.1 实验设置

论文在三个多跳问答数据集上评测DR-RAG的有效性:

- HotpotQA:基于Wikipedia的多段落问答数据集,要求根据多个文档的信息推理出答案。
- 2Wiki:一个新构建的中文多跳问题数据集,包含从百度百科中抽取的文档。
- MuSiQue:一个更具挑战性的多步骤问答数据集,涉及多个推理步骤和多文档交互。

评价指标包括准确率(Acc)、完全匹配(EM)、F1值(F1)。此外,还统计了模型推理次数(Step)和每个问题的平均用时(Time)。

### 3.2 实验结果

表1展示了DR-RAG相比现有方法在三个数据集上的效果提升:

![https://cdn.nlark.com/yuque/0/2024/png/406504/1718670910538-22f6f4e8-692d-4d73-bf6d-39c67bda1dcb.png](https://cdn.nlark.com/yuque/0/2024/png/406504/1718670910538-22f6f4e8-692d-4d73-bf6d-39c67bda1dcb.png)

可以看到,DR-RAG在三个数据集上的EM值、F1值和Acc值均取得最佳,同时推理步骤和时间开销也低于大部分基线方法。这证明了该方法在回答复杂问题上的有效性和效率。

此外,通过消融实验分别验证了QDC、CIS和CFS三个模块的有效性。表2展示了它们在MuSiQue数据集上的效果:

![https://cdn.nlark.com/yuque/0/2024/png/406504/1718670941883-436646f5-b951-4e91-9e8e-bdb76e6ed1ef.png](https://cdn.nlark.com/yuque/0/2024/png/406504/1718670941883-436646f5-b951-4e91-9e8e-bdb76e6ed1ef.png)

可以看到,无论检索的文档数量如何变化,CFS策略始终取得最佳的EM值,这说明了动态文档筛选的有效性。即使在资源受限无法训练分类器时,单独使用QDC也能在一定程度上提升检索质量。

### 3.3 案例分析

![https://files.mdnice.com/user/18983/b6a7c19e-d1b0-42e1-8f42-9be3b9f0ecee.png](https://files.mdnice.com/user/18983/b6a7c19e-d1b0-42e1-8f42-9be3b9f0ecee.png)

![https://cdn.nlark.com/yuque/0/2024/png/406504/1718670956961-75c39991-d3f5-4cbe-ba7e-35f6559ec58e.png](https://cdn.nlark.com/yuque/0/2024/png/406504/1718670956961-75c39991-d3f5-4cbe-ba7e-35f6559ec58e.png)

图3展示了一个DR-RAG在HotpotQA数据集上成功回答多跳问题的实例。可以看到,单独使用原始查询,只能检索到片面的信息,而通过QDC方法,加入了关键线索"Bob Eggleton",从而成功定位到包含答案"1960年"的动态相关文档。这直观地体现了DR-RAG利用文档间动态相关性的独特优势。

# 四、总结与展望

DR-RAG是一种新颖的检索增强生成框架,通过挖掘文档间的动态相关性,来提高复杂问题的answering能力。其主要贡献包括:

1. 提出了一种高效的两阶段检索策略,通过查询-文档拼接,挖掘隐含的相关文档。
2. 引入轻量级分类器,通过前向/反向筛选,消除冗余信息。
3. 在三个公开数据集上的实验表明,DR-RAG在准确性和效率方面都优于现有方法。

未来,DR-RAG还可以从以下几个方面进一步改进:

1. 在更多垂直领域数据上验证方法的泛化能力。
2. 优化分类器的设计,如采用对比学习等高效训练范式。
3. 利用知识蒸馏等技术,进一步压缩模型,提高推理速度。

DR-RAG为检索增强大模型在知识密集型任务上的应用提供了新的思路。随着研究的不断深入,相信类似方法能够让LLMs真正成为智能问答系统的得力助手。