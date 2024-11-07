## Embedding的benchmark
### 评估榜单
> <font style="color:rgb(0, 0, 0);">MTEB: Massive Text Embedding Benchmark</font>
>

MTEB：2023年5月，Hugging Face和Cohere.ai联合发布Embedding的benchmark

C-MTEB<font style="color:rgb(0, 0, 0);">：2023年8月，智谱和Hugging Face发布的中文评测Benchmark</font>

<font style="color:rgb(0, 0, 0);">leaderboard：</font>[https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

![](https://cdn.nlark.com/yuque/0/2024/png/406504/1705978864341-5d9c58db-7e87-4db0-aa99-38d7a9465018.png)

### <font style="color:rgb(0, 0, 0);">评估任务（MTEB定义）</font>
![](https://cdn.nlark.com/yuque/0/2024/png/406504/1705977467800-d567ae1a-5668-4634-8fae-cf9e36783054.png)

**<font style="color:rgb(63, 63, 63);">BitextMining(双语语料库挖掘)</font>**<font style="color:rgb(63, 63, 63);">：输入是来自两种不同语言的两组句子。对于第一组中的每个句子，需要在第二组中找到最佳匹配。匹配通常为翻译。提供的模型用于嵌入每句话，并通过余弦相似度查找最近的一对。F1 作为双语语料库挖掘的主要指标。还计算了准确率、精确率和召回率。</font>

**<font style="color:rgb(63, 63, 63);">Classification(分类)</font>**<font style="color:rgb(63, 63, 63);">：通过提供模型，生成训练集和测试集嵌入。使用训练集嵌入来训练逻辑回归分类器，最大迭代次数为100，该分类器在测试集上进行评分。主要指标是准确率、平均精度和f1值。</font>

**<font style="color:rgb(63, 63, 63);">Clustering(聚类)</font>**<font style="color:rgb(63, 63, 63);">：给定一组句子或段落，目标是将它们分组为有意义的群集。在嵌入文本上训练了一个小批量k-means模型（Pedregosa等人，2011年），批大小为32，k等于不同标签的数量。使用v-measure（Rosenberg和Hirschberg，2007）对模型进行评分。v-measure不依赖于聚类标签，因此标签的排列不会影响分数。</font>

**<font style="color:rgb(63, 63, 63);">PairClassification(配对分类)</font>**<font style="color:rgb(63, 63, 63);">：输入两段文本，需要分配一个标签。 标签通常是二进制变量，表示重复或近义词对。 这两个文本被嵌入并使用各种度量（余弦相似度、点积、欧几里得距离、曼哈顿距离）计算它们之间的距离。 使用最佳二元阈值精度、平均精度、F1、精确率和召回率进行计算。 基于余弦相似性的平均精度得分是主要指标。</font>

**<font style="color:rgb(63, 63, 63);">Reranking(重排序)：</font>**<font style="color:rgb(63, 63, 63);">输入是一组查询和一组相关或不相关的参考文本。目的是根据它们与查询的相关性对结果进行排名。模型用于嵌入参考文献，然后使用余弦相似度将其与查询进行比较。为每个查询生成一个排名，并对其所有查询取平均值。指标包括均值MRR@k和MAP，其中后者是主要指标。</font>

**<font style="color:rgb(63, 63, 63);">Retrieval(检索)：</font>**<font style="color:rgb(63, 63, 63);">每个数据集都包含一个语料库、查询和每个查询到语料库中相关文档的映射。目标是找到这些相关的文档。提供的模型用于嵌入所有查询和语料库中的所有文档，并使用余弦相似度计算相似性分数。根据分数对每个查询的语料库文档进行排名后，对于几个值k，可以计算nDCG@k、MRR@k、MAP@k、precision@k和recall@k。nDCG@10作为主要指标。MTEB重用BEIR（Thakur等人，2021）的数据集和评估。</font>

**<font style="color:rgb(63, 63, 63);">Semantic Textual Similarity(语义文本相似度（STS）)：</font>**<font style="color:rgb(63, 63, 63);">给定一对句子，目的是确定它们之间的相似性。标签是连续得分，数值越高表示越相似的句子。提供的模型用于嵌入句子，并使用各种距离度量计算其相似性。通过皮尔逊相关系数和斯皮尔曼等级相关系数与地面真实相似性进行比较。基于余弦相似性的斯皮尔曼等级相关系数用作主要指标（Reimers等人，2016）。</font>

**<font style="color:rgb(63, 63, 63);">Summarization(总结)</font>**<font style="color:rgb(63, 63, 63);">：提供了由人类编写的和机器生成的摘要。目的是对机器生成的摘要进行评分。首先，使用提供的模型对所有摘要进行嵌入。对于每个机器生成的摘要嵌入，计算其与所有人类生成的摘要嵌入之间的距离。保留最近的距离（例如，最高的余弦相似度），并将其用作单个机器生成的摘要的分数。计算与机器生成的摘要的地面真实评估之间的人类评估的相关性（Pearson相关性和Spearman相关性）。如STS所示，基于余弦相似性的Spearman相关性作为主要指标（Reimers等人，2016年）。</font>

> **C-MTEB中文benchmark去除了****<font style="color:rgb(63, 63, 63);">Summarization和BitextMining任务。</font>**
>

### <font style="color:rgb(0, 0, 0);">适配中文的主流Embedding模型</font>
<font style="color:rgb(0, 0, 0);">M3E：2023年6月，</font><font style="color:rgb(62, 62, 62);">Moka（北京希瑞亚斯科技）开源</font>

<font style="color:rgb(0, 0, 0);">BGE：2023年8月，清华智谱开源</font>

<font style="color:rgb(0, 0, 0);">GTE：2023年8月，阿里达摩院开源</font>

<font style="color:rgb(0, 0, 0);">Xiaobu：2023年8月，OPPO发布开源</font>

<font style="color:rgb(0, 0, 0);">PEG：2023年11月，腾讯优图实验室开源</font>

<font style="color:rgb(0, 0, 0);">tao-8k：</font>[tao-8k是由Huggingface开发者amu研发并开源的长文本向量表示模型](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/7lq0buxys)

<font style="color:rgb(0, 0, 0);">Baichuan-text-embedding：百川发布的API，付费</font>

<font style="color:rgb(0, 0, 0);">bce-embedding-base_v1：2023年12月，网易有道开源</font>

### <font style="color:rgb(0, 0, 0);">Reranker模型</font>
#### <font style="color:rgb(35, 57, 77);background-color:rgb(253, 253, 253);">为什么需要Reranker</font>
**<font style="color:rgb(35, 57, 77);background-color:rgb(253, 253, 253);">因为在搜索的时候存在随机性（ANN），这应该就是我们在RAG中第一次召回的结果往往不太满意的原因</font>**<font style="color:rgb(35, 57, 77);background-color:rgb(253, 253, 253);">。但是这也没办法，如果你的索引有数百万甚至千万的级别，那你只能牺牲一些精确度，换回时间。这时候我们可以做的就是增加</font><font style="color:rgb(35, 57, 77);background-color:rgb(238, 238, 238);">top_k</font><font style="color:rgb(35, 57, 77);background-color:rgb(253, 253, 253);">的大小，比如从原来的10个，增加到30个。然后再使用更精确的算法来做rerank，使用一一计算打分的方式，做好排序。比如30次的遍历相似度计算的时间，我们还是可以接受的。</font>

#### <font style="color:rgb(35, 57, 77);background-color:rgb(253, 253, 253);">Reranker</font><font style="color:rgb(35, 57, 77);">评测方法</font>
<font style="color:rgb(35, 57, 77);">为了衡量我们的检索系统的有效性，我们主要依赖于两个被广泛接受的指标:</font>**<font style="color:rgb(35, 57, 77);">命中率</font>**<font style="color:rgb(35, 57, 77);">和**平均倒数排名(MRR)**。让我们深入研究这些指标，了解它们的重要性以及它们是如何运作的。我们来解释一下这两个指标：</font>

+ <font style="color:rgb(35, 57, 77);">**命中率:**Hit rate计算在前k个检索文档中找到正确答案的查询比例。简单来说，它是关于我们的系统在前几次猜测中正确的频率。</font>
+ <font style="color:rgb(35, 57, 77);">**平均倒数排名(MRR):**对于每个查询，MRR通过查看排名最高的相关文档的排名来评估系统的准确性。具体来说，它是所有查询中这些秩的倒数的平均值。因此，如果第一个相关文档是顶部结果，则倒数排名为1;如果是第二个，倒数是1/2，以此类推。</font>

#### <font style="color:rgb(35, 57, 77);">Rerank模型:</font>
+ [<font style="color:rgb(35, 57, 77);background-color:rgb(253, 253, 253);">CohereAI</font>](https://txt.cohere.com/rerank/)<font style="color:rgb(35, 57, 77);background-color:rgb(253, 253, 253);">：API，收费</font>
+ <font style="color:rgb(35, 57, 77);background-color:rgb(253, 253, 253);">bge-reranker-base：智谱开源</font>
+ <font style="color:rgb(35, 57, 77);background-color:rgb(253, 253, 253);">bge-reranker-large：智谱开源</font>
+ bce-reranker-base_v1：网易有道开源

<font style="color:rgb(35, 57, 77);background-color:rgb(253, 253, 253);">有道基于llamaindex的评估leaderboard：</font>

![来源：网易有道测试结果](https://cdn.nlark.com/yuque/0/2024/png/406504/1705977165536-6d3ab416-eee1-4c6e-a849-76718d60ee1d.png)

+ WithoutReranker提供了没有重排情况下，不同Embedding模型的整体情况
+ 增加Reranker模块后，数据清楚地表明了重新排名在优化搜索结果方面的重要性。几乎所有嵌入都受益于重新排名，显示出命中率和MRR的提高

### 选择原则
+ 中英差异：同样的模型，在不同语言上效果是存在明显差异的，【[leaderboard](https://huggingface.co/spaces/mteb/leaderboard)】
+ 任务差异：同一个模型，在不同的任务上也存在一定差异性【[leaderboard](https://huggingface.co/spaces/mteb/leaderboard)】
+ 模型速度/规模：MTEB论文中进行不同参数量模型生成速度测试如下图

![](https://cdn.nlark.com/yuque/0/2024/png/406504/1705978095602-fc2c1b50-7899-4c07-9431-ccad9054acfe.png)

根据当前开源评测结果，Embedding模型：

中文：bce，模型效果好，参数量适中，不会占用太多资源

英文：uae/bge，e5效果最好，但参数量太大（14G）

Reranker模型：

中文：bge/bce的Reranker

英文：cohere/bce的Reranker

## 参考文献
[MTEB Benchmark](https://github.com/embeddings-benchmark/mteb)

[C-Pack:PackagedResourcesToAdvanceGeneralChineseEmbedding](https://arxiv.org/pdf/2309.07597.pdf)

[网易有道官方leaderboard](https://huggingface.co/maidalun1020/bce-embedding-base_v1#-leaderboard)

[llamaindex：Picking the Best Embedding & Reranker models](https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83)

[BCE基于llamaindex的rag评测指标](基于llamaindex的rag评测指标)

[中文Embedding评估实现](https://github.com/netease-youdao/BCEmbedding/tree/master/BCEmbedding)













## 
