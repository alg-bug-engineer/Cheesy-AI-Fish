# 1 RAG定义&必要性
**RAG（Retrieval Augmented Generation, 检索增强生成）**，即 LLM 在回答问题或生成文本时，先会从大量文档中检索出相关的信息，然后基于这些信息生成回答或文本，从而提高预测质量。

RAG 方法使得开发者不必为每一个特定的任务重新训练整个大模型，只需要外挂上知识库，即可为模型提供额外的信息输入，提高其回答的准确性。RAG模型尤其适合知识密集型的任务。



在 LLM 已经具备了较强能力的基础上，仍然需要 RAG ，主要有以下几点原因：

+ **幻觉问题**：LLM 文本生成的底层原理是基于概率的 token by token 的形式，因此会不可避免地产生“一本正经的胡说八道”的情况。
+ **时效性问题**：LLM 的规模越大，大模型训练的成本越高，周期也就越长。那么具有时效性的数据也就无法参与训练，所以也就无法直接回答时效性相关的问题，例如“帮我推荐几部热映的电影？”。
+ **数据安全问题**：通用的 LLM 没有企业内部数据和用户数据，那么企业想要在保证安全的前提下使用 LLM，最好的方式就是把数据全部放在本地，企业数据的业务计算全部在本地完成。而在线的大模型仅仅完成一个归纳的功能。



RAG 具有以下优点：

+ **可扩展性 (Scalability)**：减少模型大小和训练成本，并允许轻松扩展知识
+ **准确性 (Accuracy)**：模型基于事实并减少幻觉
+ **可控性 (Controllability)**：允许更新或定制知识
+ **可解释性 (Interpretability)**：检索到的项目作为模型预测中来源的参考
+ **多功能性 (Versatility)**：RAG 可以针对多种任务进行微调和定制，包括QA、文本摘要、对话系统等。

# 2 RAG方法
为了构建检索增强 LLM 系统，需要实现的关键模块和解决的问题包括:

+ **数据和索引模块**<font style="color:rgb(25, 27, 31);">: 如何处理外部数据和构建索引</font>
+ **查询和检索模块**<font style="color:rgb(25, 27, 31);">: 如何准确高效地检索出相关信息</font>
+ **响应生成模块**<font style="color:rgb(25, 27, 31);">: 如何利用检索出的相关信息来增强 LLM 的输出</font>

<font style="color:rgb(25, 27, 31);">如下图所示：</font>![](https://cdn.nlark.com/yuque/0/2023/png/35925915/1702263148635-63ff35ea-f7a7-4082-9f8c-fadc60a42620.png)

## <font style="color:rgb(25, 27, 31);">2.1 数据索引模块</font>
### 2.1.1 数据获取
数据获取模块的作用一般是将多种来源、多种类型和格式的外部数据转换成一个统一的文档对象 ( Document Object )，便于后续流程的处理和使用。文档对象除了包含原始的文本内容，一般还会携带文档的元信息 ( Metadata )，可以用于后期的检索和过滤。元信息包括但不限于：

+ 时间信息，比如文档创建和修改时间
+ 标题、关键词、实体(人物、地点等)、文本类别等信息
+ 文本总结和摘要

<font style="color:rgb(25, 27, 31);">有些元信息可以直接获取，有些则可以借助 NLP 技术，比如关键词抽取、实体识别、文本分类、文本摘要等。既可以采用传统的 NLP 模型和框架，也可以基于 LLM 实现。</font>

### 2.1.2 文本分块
<font style="color:rgb(25, 27, 31);">由于文本可能较长，或者仅有部分内容相关的情况下，需要对文本进行分块切分。</font>

<font style="color:rgb(25, 27, 31);">设计分块策略时需要考虑的因素：</font>

+ **原始内容的特点**<font style="color:rgb(25, 27, 31);">：原始内容是长文 ( 博客文章、书籍等 ) 还是短文 ( 推文、即时消息等 )，是什么格式 ( HTML、Markdown、Code 还是 LaTeX 等 )，不同的内容特点可能会适用不同的分块策略；</font>
+ **后续使用的索引方法**<font style="color:rgb(25, 27, 31);">：目前最常用的索引是对分块后的内容进行向量索引，那么不同的向量嵌入模型可能有其适用的分块大小，比如 </font>**sentence-transformer**<font style="color:rgb(25, 27, 31);"> 模型比较适合对句子级别的内容进行嵌入，OpenAI 的 </font>**text-embedding-ada-002**<font style="color:rgb(25, 27, 31);"> 模型比较适合的分块大小在 256~512 个标记数量；</font>
+ **问题的长度**<font style="color:rgb(25, 27, 31);">：问题的长度需要考虑，因为需要基于问题去检索出相关的文本片段；</font>
+ **检索出的相关内容在回复生成阶段的使用方法**<font style="color:rgb(25, 27, 31);">：如果是直接把检索出的相关内容作为 Prompt 的一部分提供给 LLM，那么 LLM 的输入长度限制在设计分块大小时就需要考虑。</font>

#### <font style="color:rgb(25, 27, 31);">2.1.2.1 分块实现方法</font>
一般来说，实现文本分块的方式如下:

+ <font style="color:rgb(25, 27, 31);">固定大小的分块方式：一般是256/512个tokens，取决于embedding模型的情况，弊端是会损失很多语义。</font>
+ <font style="color:rgb(25, 27, 31);">基于意图的分块方式：</font>
    - **<font style="color:rgb(25, 27, 31);">句分割</font>**<font style="color:rgb(25, 27, 31);">：将原始的长文本切分成小的语义单元，这里的语义单元通常是句子级别或者段落级别；</font>
    - **<font style="color:rgb(25, 27, 31);">递归分割</font>**<font style="color:rgb(25, 27, 31);">：迭代构建下一个文本片段，一般相邻的文本片段之间会设置重叠，以保持语义的连贯性。</font>

**<font style="color:rgb(25, 27, 31);">有效分块能够促进检索系统精确定位相关上下文段落以生成响应。这些块的质量和结构对于系统的功效至关重要，确保检索到的文本针对用户的查询进行精确定制。</font>**

+ <font style="color:rgb(25, 27, 31);">分块显著影响生成内容质量</font>
+ <font style="color:rgb(25, 27, 31);">较大的分块窗口效果较好，到那时超过某个最佳窗口后就开始减弱</font>
+ <font style="color:rgb(25, 27, 31);">虽然较大的块大小可以提高性能，但过多的上下文可能会引入噪音</font>

### 2.1.3 数据索引
<font style="color:rgb(25, 27, 31);">索引是一种数据结构，用于快速检索出与用户查询相关的文本内容。它是RAG的核心基础组件之一。以下引用</font><font style="color:rgb(25, 27, 31);">LlamaIndex</font><font style="color:rgb(25, 27, 31);">的介绍。</font>

#### <font style="color:rgb(25, 27, 31);">2.1.3.1 链式索引</font>
<font style="color:rgb(25, 27, 31);">链式索引通过链表的结构对文本块进行顺序索引。在后续的检索和生成阶段，可以简单地顺序遍历所有节点，也可以基于关键词进行过滤。</font>![](https://cdn.nlark.com/yuque/0/2023/webp/35925915/1702278753198-984cf085-15e7-40fb-861f-96ff4a735366.webp)

#### 2.1.3.2 **<font style="color:rgb(25, 27, 31);">树索引</font>**
<font style="color:rgb(25, 27, 31);">树索引将一组节点 ( 文本块 ) 构建成具有层级的树状索引结构，其从叶节点 (原始文本块) 向上构建，每个父节点都是子节点的摘要。在检索阶段，既可以从根节点向下进行遍历，也可以直接利用根节点的信息。树索引提供了一种更高效地查询长文本块的方式，它还可以用于从文本的不同部分提取信息。与链式索引不同，树索引无需按顺序查询。</font>

![](https://cdn.nlark.com/yuque/0/2023/png/35925915/1702278814636-ebd1e881-c939-4a41-aa33-4cf917d9a9c6.png)

#### <font style="color:rgb(25, 27, 31);">2.1.3.3 </font>**<font style="color:rgb(25, 27, 31);">关键词表索引</font>**
<font style="color:rgb(25, 27, 31);">关键词表索引从每个节点中提取关键词，构建了每个关键词到相应节点的多对多映射，意味着每个关键词可能指向多个节点，每个节点也可能包含多个关键词。在检索阶段，可以基于用户查询中的关键词对节点进行筛选。</font>

![](https://cdn.nlark.com/yuque/0/2023/png/35925915/1702278874390-98c698d0-62e9-483e-9e94-3c16facc489b.png)

![](https://cdn.nlark.com/yuque/0/2023/png/35925915/1702278922794-706616d9-a88a-4dda-8ca4-b77b45b3eacb.png)

#### 2.1.3.4 向量索引
<font style="color:rgb(25, 27, 31);">向量索引是当前最流行的一种索引方法。这种方法一般利用</font>**<font style="color:rgb(25, 27, 31);">文本嵌入模型 </font>**<font style="color:rgb(25, 27, 31);">( Text Embedding Model ) 将文本块映射成一个固定长度的向量，然后存储在</font>**<font style="color:rgb(25, 27, 31);">向量数据库</font>**<font style="color:rgb(25, 27, 31);">中。检索的时候，对用户查询文本采用同样的文本嵌入模型映射成向量，然后基于向量相似度计算获取最相似的一个或者多个节点。</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/35925915/1702278988639-683b1610-1f74-4234-9e6f-7859e39f6dd3.webp)

## <font style="color:rgb(25, 27, 31);">2.2 </font>**查询和检索模块**
### 2.2.1 查询变换
直接用原始的查询文本进行检索在很多时候可能是简单有效的，但有时候可能需要对查询文本进行一些变换，以得到更好的检索结果，从而更可能在后续生成更好的回复结果。下面列出几种常见的查询变换方式。

#### 2.2.1.1 变换一: 同义改写
将原始查询改写成相同语义下不同的表达方式，改写工作可以调用 LLM 完成。比如对于这样一个原始查询: `What are the approaches to Task Decomposition?`，可以改写成下面几种同义表达:

+ How can Task Decomposition be approached?
+ What are the different methods for Task Decomposition?
+ What are the various approaches to decomposing tasks?

对于每种查询表达，分别检索出一组相关文档，然后对所有检索结果进行去重合并，从而得到一个更大的候选相关文档集合。通过将同一个查询改写成多个同义查询，能够克服单一查询的局限，获得更丰富的检索结果集合。

#### 2.2.1.2 变换二: 查询分解
有相关研究表明 ( [self-ask](https://ofir.io/self-ask.pdf)，[ReAct](https://arxiv.org/abs/2210.03629) )，LLM 在回答复杂问题时，如果将复杂问题分解成相对简单的子问题，回复表现会更好。这里又可以分成**单步分解**和**多步分解**。

**单步分解**将一个复杂查询转化为多个简单的子查询，融合每个子查询的答案作为原始复杂查询的回复。

![](https://cdn.nlark.com/yuque/0/2023/png/35925915/1702279370838-bbbdcc7b-9d77-4a4e-8254-d853ec849fb4.png)

<font style="color:rgb(25, 27, 31);">对于</font>**<font style="color:rgb(25, 27, 31);">多步分解</font>**<font style="color:rgb(25, 27, 31);">，给定初始的复杂查询，会一步一步地转换成多个子查询，结合前一步的回复结果生成下一步的查询问题，直到问不出更多问题为止。最后结合每一步的回复生成最终的结果。</font>

![](https://cdn.nlark.com/yuque/0/2023/png/35925915/1702279456118-83bb39f9-4233-4beb-948b-5f2e9cb445e5.png)

#### <font style="color:rgb(25, 27, 31);">2.2.1.3 变换三</font>**<font style="color:rgb(25, 27, 31);">: HyDE</font>**
[HyDE](http://boston.lti.cs.cmu.edu/luyug/HyDE/HyDE.pdf)<font style="color:rgb(25, 27, 31);">，全称叫 Hypothetical Document Embeddings，给定初始查询，首先利用 LLM 生成一个假设的文档或者回复，然后以这个假设的文档或者回复作为新的查询进行检索，而不是直接使用初始查询。下面是论文中给出的一个例子:</font>

![](https://cdn.nlark.com/yuque/0/2023/png/35925915/1702279651810-8a90fdbb-3662-45a3-bc29-653f39732fa8.png)

### 2.2.2 排序和后处理
经过前面的检索过程可能会得到很多相关文档，就需要进行筛选和排序。常用的筛选和排序策略包括：

+ 基于相似度分数进行过滤和排序
+ 基于关键词进行过滤，比如限定包含或者不包含某些关键词
+ 让 LLM 基于返回的相关文档及其相关性得分来重新排序
+ 基于时间进行过滤和排序，比如只筛选最新的相关文档
+ 基于时间对相似度进行加权，然后进行排序和筛选

## <font style="color:rgb(25, 27, 31);">2.3 </font>**<font style="color:rgb(25, 27, 31);">回复生成模块</font>**
### <font style="color:rgb(25, 27, 31);">2.3.1 回复生成策略</font>
<font style="color:rgb(25, 27, 31);">检索模块基于用户查询检索出相关的文本块，回复生成模块让 LLM 利用检索出的相关信息来生成对原始查询的回复。LlamaIndex 中有给出一些不同的回复生成策略。</font>

<font style="color:rgb(25, 27, 31);">一种策略是依次结合每个检索出的相关文本块，每次不断修正生成的回复。这样的话，有多少个独立的相关文本块，就会产生多少次的 LLM 调用。另一种策略是在每次 LLM 调用时，尽可能多地在 Prompt 中填充文本块。如果一个 Prompt 中填充不下，则采用类似的操作构建多个 Prompt，多个 Prompt 的调用可以采用和前一种相同的回复修正策略。</font>

### <font style="color:rgb(25, 27, 31);">2.3.2 回复生成 Prompt 模板</font>
<font style="color:rgb(25, 27, 31);">下面是 LlamaIndex 中提供的一个生成回复的 Prompt 模板。从这个模板中可以看到，可以用一些分隔符 ( 比如 ------ ) 来区分相关信息的文本，还可以指定 LLM 是否需要结合它自己的知识来生成回复，以及当提供的相关信息没有帮助时，要不要回复等。</font>

```plain
template = f'''
Context information is below.
---------------------
{context_str}
---------------------
Using both the context information and also using your own knowledge, answer the question: {query_str}

If the context isn't helpful, you can/don’t answer the question on your own.
'''
```

<font style="color:rgb(25, 27, 31);">下面的 Prompt 模板让 LLM 不断修正已有的回复。</font>

```plain
template = f'''
The original question is as follows: {query_str}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer (only if needed) with some more context below.
------------
{context_str}
------------
Using both the new context and your own knowledege, update or repeat the existing answer.
'''
```

# 3 关于检索问答的思考&改进点
## 3.1 存在的问题&改进点
### 3.1.1 检索相关性不高的问题
**改进点1**：文本分块

原因：当前检索粒度是文本段落，可能会导致无效信息过多，影响向量相似度计算

改进方式：对当前建库时的文本分块长度进行统计，**按照一定长度范围对文本进行分割**，此处需要进行实验，比较不同的长度范围对结果的影响

**改进点2**：混合检索方式

原因：当前仅用向量检索的方式，会有误召的情况，且调整相似度阈值较高的话，又会砍掉较为相关的内容，调优困难

改进方式：采用基于规则和关键词的方式进行检索，鉴于检索知识库过大的情况，可以用基于规则和关键词的方式对检索结果进行过滤，或者重新排序，以提高检索结果集合的相似度。

### 3.2.2 生成幻觉问题
**改进点1**：prompt优化

改进方式：在第一次生成答案后，设计检测幻觉的prompt，以要求模型对自己的答案进行修复。

改进点2：相关文本优化

原因：当前策略中，标题检索的结果中不包含相关文本，同时内容检索的文本中无关信息可能较多，会给模型更多噪声输入

改进方式：对标题检索出的文本进行摘要或者检索出最相关的段落，对内容检索的结果从建库时进行更细致的分块，以减少无关信息。

# 4 参考材料
1. 大模型检索增强生成（RAG）有哪些好用的技巧？ ——[https://www.zhihu.com/question/625481187/answer/3309968693](https://www.zhihu.com/question/625481187/answer/3309968693)
2. 从 RAG 到 Self-RAG —— LLM 的知识增强 —— [https://zhuanlan.zhihu.com/p/661465330](https://zhuanlan.zhihu.com/p/661465330)
3. 浅谈我对RAG的看法 —— [https://zhuanlan.zhihu.com/p/666233533](https://zhuanlan.zhihu.com/p/666233533)
4. LLM RAG界最强检索方式: L1 Hybrid Retrieval+L2 Semantic Ranking —— [https://mp.weixin.qq.com/s/yKknJQi8Yu0XlquBuufA2Q](https://mp.weixin.qq.com/s/yKknJQi8Yu0XlquBuufA2Q)
5. 如何避免大语言模型绕过知识库乱答的情况？LlamaIndex 原理与应用简介 —— [https://zhuanlan.zhihu.com/p/660246001](https://zhuanlan.zhihu.com/p/660246001)
6. 万字长文总结检索增强 LLM —— [https://zhuanlan.zhihu.com/p/655272123?utm_id=0](https://zhuanlan.zhihu.com/p/655272123?utm_id=0)



