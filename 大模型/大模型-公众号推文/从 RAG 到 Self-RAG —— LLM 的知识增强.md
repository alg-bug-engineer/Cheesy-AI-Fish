## <font style="color:rgb(25, 27, 31);">一、 RAG 及其必要性</font>
### <font style="color:rgb(25, 27, 31);">1.1 初识 RAG</font>
<font style="color:rgb(25, 27, 31);">RAG（Retrieval Augmented Generation, 检索增强生成），即 LLM 在回答问题或生成文本时，先会从大量文档中检索出相关的信息，然后基于这些信息生成回答或文本，从而提高预测质量。RAG 方法使得开发者不必为每一个特定的任务重新训练整个大模型，只需要外挂上知识库，即可为模型提供额外的信息输入，提高其回答的准确性。RAG模型尤其适合知识密集型的任务。</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292737120-796f0192-d712-43ad-b461-90d958cda89d.webp)

<font style="color:rgb(145, 150, 161);">LLM V.S. RAG LLM</font>

<font style="color:rgb(25, 27, 31);">在 LLM 已经具备了较强能力的基础上，仍然需要 RAG ，主要有以下几点原因：</font>

+ **<font style="color:rgb(25, 27, 31);">幻觉问题</font>**<font style="color:rgb(25, 27, 31);">：LLM 文本生成的底层原理是基于概率的 token by token 的形式，因此会不可避免地产生“一本正经的胡说八道”的情况。</font>
+ **<font style="color:rgb(25, 27, 31);">时效性问题</font>**<font style="color:rgb(25, 27, 31);">：LLM 的规模越大，大模型训练的成本越高，周期也就越长。那么具有时效性的数据也就无法参与训练，所以也就无法直接回答时效性相关的问题，例如“帮我推荐几部热映的电影？”。</font>
+ **<font style="color:rgb(25, 27, 31);">数据安全问题</font>**<font style="color:rgb(25, 27, 31);">：通用的 LLM 没有企业内部数据和用户数据，那么企业想要在保证安全的前提下使用 LLM，最好的方式就是把数据全部放在本地，企业数据的业务计算全部在本地完成。而在线的大模型仅仅完成一个归纳的功能。</font>

### <font style="color:rgb(25, 27, 31);">1.2 RAG V.S. SFT</font>
<font style="color:rgb(25, 27, 31);">实际上，对于 LLM 存在的上述问题，SFT 是一个最常见最基本的解决办法，也是 LLM 实现应用的基础步骤。那么有必要在多个维度上比较一下两种方法：</font>

| | **<font style="color:rgb(25, 27, 31);">RAG</font>** | **<font style="color:rgb(25, 27, 31);">SFT</font>** |
| --- | :--- | :--- |
| <font style="color:rgb(25, 27, 31);">Data</font> | <font style="color:rgb(25, 27, 31);">动态数据。 RAG 不断查询外部源，确保信息保持最新，而无需频繁的模型重新训练。</font> | <font style="color:rgb(25, 27, 31);">(相对)静态数据，并且在动态数据场景中可能很快就会过时。 SFT 也不能保证记住这些知识。</font> |
| <font style="color:rgb(25, 27, 31);">External Knowledge</font> | <font style="color:rgb(25, 27, 31);">RAG 擅长利用外部资源。通过在生成响应之前从知识源检索相关信息来增强 LLM 能力。 它非常适合文档或其他结构化/非结构化数据库。</font> | <font style="color:rgb(25, 27, 31);">SFT 可以对 LLM 进行微调以对齐预训练学到的外部知识，但对于频繁更改的数据源来说可能不太实用。</font> |
| <font style="color:rgb(25, 27, 31);">Model Customization</font> | <font style="color:rgb(25, 27, 31);">RAG 主要关注信息检索，擅长整合外部知识，但可能无法完全定制模型的行为或写作风格。</font> | <font style="color:rgb(25, 27, 31);">SFT 允许根据特定的语气或术语调整LLM 的行为、写作风格或特定领域的知识。</font> |
| <font style="color:rgb(25, 27, 31);">Reducing Hallucinations</font> | <font style="color:rgb(25, 27, 31);">RAG 本质上不太容易产生幻觉，因为每个回答都建立在检索到的证据上。</font> | <font style="color:rgb(25, 27, 31);">SFT 可以通过将模型基于特定领域的训练数据来帮助减少幻觉。 但当面对不熟悉的输入时，它仍然可能产生幻觉。</font> |
| <font style="color:rgb(25, 27, 31);">Transparency</font> | <font style="color:rgb(25, 27, 31);">RAG 系统通过将响应生成分解为不同的阶段来提供透明度，提供对数据检索的匹配度以提高对输出的信任。</font> | <font style="color:rgb(25, 27, 31);">SFT 就像一个黑匣子，使得响应背后的推理更加不透明。</font> |
| <font style="color:rgb(25, 27, 31);">Technical Expertise</font> | <font style="color:rgb(25, 27, 31);">RAG 需要高效的检索策略和大型数据库相关技术。另外还需要保持外部数据源集成以及数据更新。</font> | <font style="color:rgb(25, 27, 31);">SFT 需要准备和整理高质量的训练数据集、定义微调目标以及相应的计算资源。</font> |


<font style="color:rgb(25, 27, 31);">当然这两种方法并非非此即彼的，合理且必要的方式是结合业务需要与两种方法的优点，合理使用两种方法。</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292737121-f56a2205-ed6b-426b-9c13-a8d97501c0ee.webp)

<font style="color:rgb(145, 150, 161);">LLM, SFT, RAG 之间的关系</font>

<font style="color:rgb(25, 27, 31);">通过以上讨论，也能总结出 RAG 具有一下优点：</font>

+ **<font style="color:rgb(25, 27, 31);">可扩展性 (Scalability)</font>**<font style="color:rgb(25, 27, 31);">：减少模型大小和训练成本，并允许轻松扩展知识</font>
+ **<font style="color:rgb(25, 27, 31);">准确性 (Accuracy)</font>**<font style="color:rgb(25, 27, 31);">：模型基于事实并减少幻觉</font>
+ **<font style="color:rgb(25, 27, 31);">可控性 (Controllability)</font>**<font style="color:rgb(25, 27, 31);">：允许更新或定制知识</font>
+ **<font style="color:rgb(25, 27, 31);">可解释性 (Interpretability)</font>**<font style="color:rgb(25, 27, 31);">：检索到的项目作为模型预测中来源的参考</font>
+ **<font style="color:rgb(25, 27, 31);">多功能性 (Versatility)</font>**<font style="color:rgb(25, 27, 31);">：RAG 可以针对多种任务进行微调和定制，包括QA、文本摘要、对话系统等。</font>

## <font style="color:rgb(25, 27, 31);">二、RAG 典型方法及案例</font>
<font style="color:rgb(25, 27, 31);">上节简要介绍了 RAG 及其基本特点，本节将详细介绍 RAG 的实现及其应用。</font>

### <font style="color:rgb(25, 27, 31);">2.1 RAG 典型实现方法</font>
<font style="color:rgb(25, 27, 31);">概括来说，RAG 的实现主要包括三个主要步骤：数据索引、检索和生成。如下图所示：</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292737153-bf3b8169-4ec5-4301-be70-e3a570160340.webp)

**<font style="color:rgb(25, 27, 31);">2.1.1 数据索引的构建</font>**

<font style="color:rgb(25, 27, 31);">该部分主要功能是将原始数据处理成为便于检索的格式（通常为embedding），该过程又可以进一步分为：</font>

**<font style="color:rgb(25, 27, 31);">Step1: 数据提取</font>**

<font style="color:rgb(25, 27, 31);">即从原始数据到便于处理的格式化数据的过程，具体工程包括：</font>

+ <font style="color:rgb(25, 27, 31);">数据获取：获得作为知识库的多种格式的数据，包括PDF、word、markdown以及数据库和API等；</font>
+ <font style="color:rgb(25, 27, 31);">数据清洗：对源数据进行去重、过滤、压缩和格式化等处理；</font>
+ <font style="color:rgb(25, 27, 31);">信息提取：提取重要信息，包括文件名、时间、章节title、图片等信息。</font>

**<font style="color:rgb(25, 27, 31);">Step 2: 分块（Chunking）</font>**

<font style="color:rgb(25, 27, 31);">由于文本可能较长，或者仅有部分内容相关的情况下，需要对文本进行分块切分，分块的方式有：</font>

+ <font style="color:rgb(25, 27, 31);">固定大小的分块方式：一般是256/512个tokens，取决于embedding模型的情况，弊端是会损失很多语义。</font>
+ <font style="color:rgb(25, 27, 31);">基于意图的分块方式：</font>
    - <font style="color:rgb(25, 27, 31);">句分割：最简单的是通过句号和换行来做切分，常用的意图包有基于NLP的NLTK和spaCy；</font>
    - <font style="color:rgb(25, 27, 31);">递归分割：通过分而治之的思想，用递归切分到最小单元的一种方式；</font>
    - <font style="color:rgb(25, 27, 31);">特殊分割：用于特殊场景。</font>

<font style="color:rgb(25, 27, 31);">常用的工具如 langchain.text_splitter 库中的类CharacterTextSplitter，可以指定分隔符、块大小、重叠和长度函数来拆分文本。</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292737152-19c972eb-585b-40c5-adb4-fbf8c22543e5.webp)

**<font style="color:rgb(25, 27, 31);">Step 3: 向量化（embedding）及创建索引</font>**

<font style="color:rgb(25, 27, 31);">即将文本、图像、音频和视频等转化为向量矩阵的过程，也就是变成计算机可以理解的格式。常用的工具如</font><font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">langchain.embeddings.openai.OpenAIEmbeddings</font><font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">和</font><font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">langchain.embeddings.huggingface.HuggingFaceEmbeddings</font><font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">等。</font>

<font style="color:rgb(25, 27, 31);">生成 embedding 之后就是创建索引。最常见的即使用 FAISS 库创建向量搜索索引。使用</font><font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">langchain.vectorstores</font><font style="color:rgb(25, 27, 31);">库中的 FAISS 类的</font><font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">from_texts</font><font style="color:rgb(25, 27, 31);">方法，使用文本块和生成的嵌入来构建搜索索引。</font>



```plain
docsearch = FAISS.from_texts(texts, embeddings)
```

**<font style="color:rgb(25, 27, 31);">2.1.2 检索（Retrieval）</font>**

<font style="color:rgb(25, 27, 31);">检索环节是获取有效信息的关键环节。检索优化一般分为下面五部分工作：</font>

+ **<font style="color:rgb(25, 27, 31);">元数据过滤</font>**<font style="color:rgb(25, 27, 31);">：当我们把索引分成许多chunks的时候，检索效率会成为问题。这时候，如果可以通过元数据先进行过滤，就会大大提升效率和相关度。</font>
+ **<font style="color:rgb(25, 27, 31);">图关系检索</font>**<font style="color:rgb(25, 27, 31);">：即引入知识图谱，将实体变成node，把它们之间的关系变成relation，就可以利用知识之间的关系做更准确的回答。特别是针对一些多跳问题，利用图数据索引会让检索的相关度变得更高；</font>
+ **<font style="color:rgb(25, 27, 31);">检索技术</font>**<font style="color:rgb(25, 27, 31);">：检索的主要方式还是这几种：</font>
    - **<font style="color:rgb(25, 27, 31);">相似度检索</font>**<font style="color:rgb(25, 27, 31);">：包括欧氏距离、曼哈顿距离、余弦等；</font>
    - **<font style="color:rgb(25, 27, 31);">关键词检索</font>**<font style="color:rgb(25, 27, 31);">：这是很传统的检索方式，元数据过滤也是一种，还有一种就是先把chunk做摘要，再通过关键词检索找到可能相关的chunk，增加检索效率。</font>
    - **<font style="color:rgb(25, 27, 31);">SQL检索</font>**<font style="color:rgb(25, 27, 31);">：更加传统的检索算法。</font>
+ **<font style="color:rgb(25, 27, 31);">重排序（Rerank）</font>**<font style="color:rgb(25, 27, 31);">：相关度、匹配度等因素做一些重新调整，得到更符合业务场景的排序。</font>
+ **<font style="color:rgb(25, 27, 31);">查询轮换</font>**<font style="color:rgb(25, 27, 31);">：这是查询检索的一种方式，一般会有几种方式：</font>
    - **<font style="color:rgb(25, 27, 31);">子查询：</font>**<font style="color:rgb(25, 27, 31);">可以在不同的场景中使用各种查询策略，比如可以使用LlamaIndex等框架提供的查询器，采用树查询（从叶子结点，一步步查询，合并），采用向量查询，或者最原始的顺序查询chunks等</font>**<font style="color:rgb(25, 27, 31);">；</font>**
    - **<font style="color:rgb(25, 27, 31);">HyDE：</font>**<font style="color:rgb(25, 27, 31);">这是一种抄作业的方式，生成相似的或者更标准的 prompt 模板</font>**<font style="color:rgb(25, 27, 31);">。</font>**

**<font style="color:rgb(25, 27, 31);">2.1.3 文本生成</font>**

<font style="color:rgb(25, 27, 31);">文本生成就是将原始 query 和检索得到的文本组合起来输入模型得到结果的过程，本质上就是个 prompt engineer ing 过程，可参考笔者之前的文章</font>

[紫气东来：NLP（十三）：Prompt Engineering 面面观36 赞同 · 2 评论文章](https://zhuanlan.zhihu.com/p/632369186)

<font style="color:rgb(25, 27, 31);">此外还有全流程的框架，如</font><font style="color:rgb(25, 27, 31);"> </font>[Langchain](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/use_cases/question_answering/)<font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">和</font><font style="color:rgb(25, 27, 31);"> </font>[LlamaIndex](https://link.zhihu.com/?target=https%3A//gpt-index.readthedocs.io/en/latest/examples/index_structs/knowledge_graph/KnowledgeGraphIndex_vs_VectorStoreIndex_vs_CustomIndex_combined.html)<font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">，都非常简单易用，如：</font>



```plain
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | rag_prompt | llm
rag_chain.invoke("What is Task Decomposition?")
```

### <font style="color:rgb(25, 27, 31);">2.2 RAG 典型案例</font>
<font style="color:rgb(25, 27, 31);">2.2.1</font><font style="color:rgb(25, 27, 31);"> </font>[ChatPDF](https://link.zhihu.com/?target=https%3A//www.chatpdf.com/)<font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">及其</font><font style="color:rgb(25, 27, 31);"> </font>[复刻版](https://link.zhihu.com/?target=https%3A//github.com/shibing624/ChatPDF)

<font style="color:rgb(25, 27, 31);">ChatPDF的实现流程如下：</font>

1. <font style="color:rgb(25, 27, 31);">ChatPDF首先读取PDF文件，将其转换为可处理的文本格式，例如txt格式。</font>
2. <font style="color:rgb(25, 27, 31);">接着，ChatPDF会对提取出来的文本进行清理和标准化，例如去除特殊字符、分段、分句等，以便于后续处理。这一步可以使用自然语言处理技术，如正则表达式等。</font>
3. <font style="color:rgb(25, 27, 31);">ChatPDF使用OpenAI的Embeddings API将每个分段转换为向量，这个向量将对文本中的语义进行编码，以便于与问题的向量进行比较。</font>
4. <font style="color:rgb(25, 27, 31);">当用户提出问题时，ChatPDF使用OpenAI的Embeddings API将问题转换为一个向量，并与每个分段的向量进行比较，以找到最相似的分段。这个相似度计算可以使用余弦相似度等常见的方法进行。</font>
5. <font style="color:rgb(25, 27, 31);">ChatPDF将找到的最相似的分段与问题作为prompt，调用OpenAI的Completion API，让ChatGPT学习分段内容后，再回答对应的问题。</font>
6. <font style="color:rgb(25, 27, 31);">最后，ChatPDF会将ChatGPT生成的答案返回给用户，完成一次查询。</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292737145-af3f5a24-64ae-4503-aabc-d4d37d07fda9.webp)

<font style="color:rgb(25, 27, 31);">另外笔者之前也实现并开源过一个类似的案例，具体可参考</font>

[紫气东来：NLP（十六）：LangChain —— 自由搭建 LLM 的应用程序77 赞同 · 8 评论文章](https://zhuanlan.zhihu.com/p/636741983)

<font style="color:rgb(25, 27, 31);">2.2.2</font><font style="color:rgb(25, 27, 31);"> </font>[Baichuan](https://link.zhihu.com/?target=https%3A//www.baichuan-ai.com/home)

<font style="color:rgb(25, 27, 31);">百川大模型的搜索增强系统融合了多个模块，包括指令意图理解、智能搜索和结果增强等组件。该体系通过深入理解用户指令，精确驱动查询词的搜索，并结合大语言模型技术来优化模型结果生成的可靠性。通过这一系列协同作用，大模型实现了更精确、智能的模型结果回答，通过这种方式减少了模型的幻觉。</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292737529-ee0a03b1-4cfb-4e5f-b492-c6f45a37dc43.webp)

<font style="color:rgb(25, 27, 31);">2.2.3</font><font style="color:rgb(25, 27, 31);"> </font>[Multi-modal retrieval-based LMs](https://link.zhihu.com/?target=https%3A//cs.stanford.edu/~myasu/blog/racm3/)

<font style="color:rgb(25, 27, 31);">RA-CM3 是一个检索增强的多模态模型，其包含了一个信息检索框架来从外部存储库中获取知识，具体来说，作者首先使用预训练的 CLIP 模型来实现一个检索器（retriever），然后使用 CM3 Transformer 架构来构成一个生成器（generator），其中检索器用来辅助模型从外部存储库中搜索有关于当前提示文本中的精确信息，然后将该信息连同文本送入到生成器中进行图像合成，这样设计的模型的准确性就会大大提高。</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292737609-8f99d95a-2ad7-47bc-b673-b1202796d9aa.webp)

<font style="color:rgb(25, 27, 31);">2.2.4</font><font style="color:rgb(25, 27, 31);"> </font>[LeanDojo](https://link.zhihu.com/?target=https%3A//leandojo.org/)

<font style="color:rgb(25, 27, 31);">这是一个通过检索增强进行数学证明的案例，其中 Lean 是公式数学的编码语言。下图描述了实现过程：</font>

+ **<font style="color:rgb(25, 27, 31);">顶部右边：</font>**<font style="color:rgb(25, 27, 31);">LeanDojo从Lean中提取证明到数据库中，用来训练机器学习模型。这个流程也可以通过和Lean的证明环境进行交互后让训练好的模型来证明定理。</font>
+ **<font style="color:rgb(25, 27, 31);">顶部左边：</font>**<font style="color:rgb(25, 27, 31);">这是Lean定理</font><font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">�</font><font style="color:rgb(25, 27, 31);">:</font><font style="color:rgb(25, 27, 31);">�</font><font style="color:rgb(25, 27, 31);">⊢</font><font style="color:rgb(25, 27, 31);">gcd</font><font style="color:rgb(25, 27, 31);">⁡</font><font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">�</font><font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">�</font><font style="color:rgb(25, 27, 31);">=</font><font style="color:rgb(25, 27, 31);">�</font><font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">的证明树。在这里gcd即最大公约数。在证明定理时，我们从原始定理作为初始状态（根）开始，并重复应用策略（边）将状态分解为更简单的子状态，直到所有状态都得到解决（叶节点处）。策略可能依赖于大型数学库中定义的诸如 mod_self 和 gcd_zero_left 之类的前提。例如，mod_self 是证明中用于简化目标的现有定理 ：</font><font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);"> ( </font><font style="color:rgb(25, 27, 31);">n</font><font style="color:rgb(25, 27, 31);">:</font><font style="color:rgb(25, 27, 31);"> nat) : </font><font style="color:rgb(25, 27, 31);">n</font><font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">%</font><font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">n</font>
+ **<font style="color:rgb(25, 27, 31);">底部：</font>**<font style="color:rgb(25, 27, 31);">只要给定一个状态，Reprover模型就能从数学库中检索前提，这些前提与状态连接起来，输入到一个作为编码器和解码器的Transformer中以生成下一个策略。</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292737679-faff3ea3-392e-4668-8907-070f2e529055.webp)

### <font style="color:rgb(25, 27, 31);">2.3 RAG 的挑战及部分改进方法</font>
<font style="color:rgb(25, 27, 31);">尽管 RAG 在很多例子中取得了很好的效果，仍然面临着不少挑战，主要体现在以下方面：</font>

    - <font style="color:rgb(25, 27, 31);">检索效果依赖 embedding 和检索算法，目前可能检索到无关信息，反而对输出有负面影响；</font>
    - <font style="color:rgb(25, 27, 31);">大模型如何利用检索到的信息仍是黑盒的，可能仍存在不准确（甚至生成的文本与检索信息相冲突）；</font>
    - <font style="color:rgb(25, 27, 31);">对所有任务都无差别检索 k 个文本片段，效率不高，同时会大大增加模型输入的长度；</font>
    - <font style="color:rgb(25, 27, 31);">无法引用来源，也因此无法精准地查证事实，检索的真实性取决于数据源及检索算法。</font>

<font style="color:rgb(25, 27, 31);">针对上述的问题，也逐步发展出一些改进的方法，下面仅举几例：</font>

<font style="color:rgb(25, 27, 31);">2.3.1 RAG 结合 SFT：</font>[RA-DIT](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2310.01352.pdf)

<font style="color:rgb(25, 27, 31);">简单来说，RA-DIT 方法分别对 LLM 和检索器进行微调。 更新 LLM 以最大限度地提高在给定检索增强指令的情况下正确答案的概率，同时更新检索器以最大限度地减少文档与查询在语义上相似（相关）的程度。通过这种方式，使 LLM 更好地利用相关背景知识，并训练 LLM 即使在检索错误块的情况下也能产生准确的预测，从而使模型能够依赖自己的知识。</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292737612-86b5a503-7a17-44df-8dcf-1a8294f9505b.webp)

<font style="color:rgb(25, 27, 31);">2.3.2 多向量检索器（Multi-Vector Retriever）</font>

<font style="color:rgb(25, 27, 31);">多向量检索器 (Multi-Vector Retriever) 是 LangChain 推出的一个关键工具，用于优化 RAG 的过程。多向量检索器的核心想法是将我们想要用于答案合成的文档与我们想要用于检索的参考文献分开。这允许系统为搜索优化文档的版本而不失去答案合成时的上下文。</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292737631-a0ba4a2b-cb4a-4933-8953-8f57901a1886.webp)

<font style="color:rgb(25, 27, 31);">简单来说就是将一个文档分解为较长的几个逻辑较为完整和独立的部分，例如包括不同的文本、表格甚至是图片都可以，然后分解后的文档使用摘要的方式进行总结，这个摘要需要可以明确覆盖相关内容。然后摘要进行向量化，检索的时候直接检索摘要，一旦匹配，即可将摘要背后的完整文档作为上下文输入给大模型。</font>

<font style="color:rgb(25, 27, 31);">2.3.3 查询转换（Query Transformations）</font>

<font style="color:rgb(25, 27, 31);">在某些情况下，用户的 query 可能出现表述不清、需求复杂、内容无关等问题，为了解决这些问题，查询转换（Query Transformations）的方案利用了大型语言模型(LLM)的强大能力，通过某种提示或方法将原始的用户问题转换或重写为更合适的、能够更准确地返回所需结果的查询。LLM的能力确保了转换后的查询更有可能从文档或数据中获取相关和准确的答案。</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292737944-69741e12-7ed5-4f7f-b466-2bc3492992b5.webp)

<font style="color:rgb(25, 27, 31);">查询转换的核心思想是，用户的原始查询可能不总是最适合检索的，所以我们需要某种方式来改进或扩展它。</font>

## <font style="color:rgb(25, 27, 31);">三、</font>[Self-RAG](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2310.11511)<font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">及其实现</font>
<font style="color:rgb(25, 27, 31);">前文所述的 RAG 方法都遵循着共同的范式，即：</font><font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">�</font><font style="color:rgb(25, 27, 31);">�</font><font style="color:rgb(25, 27, 31);">�</font><font style="color:rgb(25, 27, 31);">�</font><font style="color:rgb(25, 27, 31);">�</font><font style="color:rgb(25, 27, 31);">+</font><font style="color:rgb(25, 27, 31);">�</font><font style="color:rgb(25, 27, 31);">�</font><font style="color:rgb(25, 27, 31);">�</font><font style="color:rgb(25, 27, 31);">�</font><font style="color:rgb(25, 27, 31);">�</font><font style="color:rgb(25, 27, 31);">�</font><font style="color:rgb(25, 27, 31);">�</font><font style="color:rgb(25, 27, 31);">→</font><font style="color:rgb(25, 27, 31);">�</font><font style="color:rgb(25, 27, 31);">�</font><font style="color:rgb(25, 27, 31);">�</font><font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">，其中 query 表示用户的输入，context 表示检索获得的补充信息，然后共同输入到 LLM 中，可以认为这是一种检索前置的被动的增强方式。</font>

<font style="color:rgb(25, 27, 31);">相比而言，Self-RAG 则是更加主动和智能的实现方式，主要步骤概括如下：</font>

1. <font style="color:rgb(25, 27, 31);">判断是否需要额外检索事实性信息（retrieve on demand），仅当有需要时才召回</font>
2. <font style="color:rgb(25, 27, 31);">平行处理每个片段：生产prompt+一个片段的生成结果</font>
3. <font style="color:rgb(25, 27, 31);">使用反思字段，检查输出是否相关，选择最符合需要的片段；</font>
4. <font style="color:rgb(25, 27, 31);">再重复检索</font>
5. <font style="color:rgb(25, 27, 31);">生成结果会引用相关片段，以及输出结果是否符合该片段，便于查证事实。</font>

<font style="color:rgb(25, 27, 31);">二者过程的比较如下图所示：</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292738023-de0656e2-5510-4142-9e05-c3d83633c7d7.webp)

<font style="color:rgb(25, 27, 31);">Self-RAG 的一个重要创新是</font>**<font style="color:rgb(25, 27, 31);"> </font>****<font style="color:rgb(25, 27, 31);">Reflection tokens (反思字符)</font>**<font style="color:rgb(25, 27, 31);">：通过生成反思字符这一特殊标记来检查输出。这些字符会分为 Retrieve 和 Critique 两种类型，会标示：检查是否有检索的必要，完成检索后检查输出的相关性、完整性、检索片段是否支持输出的观点。模型会基于原有词库和反思字段来生成下一个 token。</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292738028-802b9223-7235-4478-9d83-567e9c65e761.webp)

### <font style="color:rgb(25, 27, 31);">3.1 Self-RAG 训练过程</font>
<font style="color:rgb(25, 27, 31);">对于训练，模型通过将反思字符集成到其词汇表中来学习生成带有反思字符的文本。 它是在一个语料库上进行训练的，其中包含由 Critic 模型预测的检索到的段落和反思字符。 该 Critic 模型评估检索到的段落和任务输出的质量。 使用反思字符更新训练语料库，并训练最终模型以在推理过程中独立生成这些字符。</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292738066-8426fe39-8b40-4862-b1c1-9221202af21e.webp)

<font style="color:rgb(25, 27, 31);">为了训练 Critic 模型，手动标记反思字符的成本很高，于是作者使用 GPT-4 生成反思字符，然后将这些知识提炼到内部 Critic 模型中。 不同的反思字符会通过少量演示来提示具体说明。 例如，检索令牌会被提示判断外部文档是否会改善结果。</font>

<font style="color:rgb(25, 27, 31);">为了训练生成模型，使用检索和 Critic 模型来增强原始输出以模拟推理过程。 对于每个片段，Critic 模型都会确定额外的段落是否会改善生成。 如果是，则添加</font><font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">Retrieve=Yes</font><font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">标记，并检索前 K 个段落。 然后 Critic 评估每段文章的相关性和支持性，并相应地附加标记。 最终通过输出反思字符进行增强。</font>

<font style="color:rgb(25, 27, 31);">然后使用标准的 next token 目标在此增强语料库上训练生成模型，预测目标输出和反思字符。 在训练期间，检索到的文本块被屏蔽，并通过反思字符 Critique 和 Retrieve 扩展词汇量。 这种方法比 PPO 等依赖单独奖励模型的其他方法更具成本效益。 Self-RAG 模型还包含特殊令牌来控制和评估其自身的预测，从而实现更精细的输出生成。</font>

### <font style="color:rgb(25, 27, 31);">3.2 Self-RAG 推理过程</font>
<font style="color:rgb(25, 27, 31);">Self-RAG 使用反思字符来自我评估输出，使其在推理过程中具有适应性。 根据任务的不同，可以定制模型，通过检索更多段落来优先考虑事实准确性，或强调开放式任务的创造力。 该模型可以决定何时检索段落或使用设定的阈值来触发检索。</font>

<font style="color:rgb(25, 27, 31);">当需要检索时，生成器同时处理多个段落，产生不同的候选。 进行片段级 beam search 以获得最佳序列。 每个细分的分数使用 Critic 分数进行更新，该分数是每个批评标记类型的归一化概率的加权和。 可以在推理过程中调整这些权重以定制模型的行为。 与其他需要额外训练才能改变行为的方法不同，Self-RAG 无需额外训练即可适应。</font>

<font style="color:rgb(25, 27, 31);">下面对开源的 Self-RAG 进行推理测试，可在这里下载模型</font><font style="color:rgb(25, 27, 31);"> </font>[selfrag_llama2_13b](https://link.zhihu.com/?target=https%3A//huggingface.co/selfrag/selfrag_llama2_13b)<font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">，按照官方指导使用 vllm 进行推理服务</font>



```plain
from vllm import LLM, SamplingParams

model = LLM("selfrag/selfrag_llama2_7b", download_dir="/gscratch/h2lab/akari/model_cache", dtype="half")
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)

def format_prompt(input, paragraph=None):
  prompt = "### Instruction:\n{0}\n\n### Response:\n".format(input)
  if paragraph is not None:
    prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
  return prompt

query_1 = "Leave odd one out: twitter, instagram, whatsapp."
query_2 = "What is China?"
queries = [query_1, query_2]

# for a query that doesn't require retrieval
preds = model.generate([format_prompt(query) for query in queries], sampling_params)
for pred in preds:
  print("Model prediction: {0}".format(pred.outputs[0].text))
```

<font style="color:rgb(25, 27, 31);">输出结果如下，其中第一段结果不需要检索，第二段结果出现</font><font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">[Retrieval]</font><font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">字段，因为这个问题需要更细粒度的事实依据。</font>



```plain
Model prediction: Twitter:[Utility:5]</s>
Model prediction: China is a country located in East Asia.[Retrieval]<paragraph>It is the most populous country in the world, with a population of over 1.4 billion people.[Retrieval]<paragraph>China is a diverse country with a rich history and culture.[Retrieval]<paragraph>It is home to many different ethnic groups and languages, and its cuisine, art, and architecture reflect this diversity.[Retrieval]<paragraph>China is also a major economic power, with a rapidly growing economy and a large manufacturing sector.[Utility:5]</s>
```

<font style="color:rgb(25, 27, 31);">我们还可以在输入中增加补充信息：</font>



```plain
# for a query that needs factual grounding
prompt = format_prompt("Can you tell me the difference between llamas and alpacas?", "The alpaca (Lama pacos) is a species of South American camelid mammal. It is similar to, and often confused with, the llama. Alpacas are considerably smaller than llamas, and unlike llamas, they were not bred to be working animals, but were bred specifically for their fiber.")
preds = model.generate([prompt], sampling_params)
print([pred.outputs[0].text for pred in preds])
```

<font style="color:rgb(25, 27, 31);">输出结果如下，Self-RAG 找到相关的插入文档并生成有证据支持的答案。</font>



```plain
['<paragraph>The main difference between llamas and alpacas is their size and fiber.[Continue to Use Evidence]Llamas are much larger than alpacas, and they have a much coarser fiber.[Utility:5]</s>']
```

### <font style="color:rgb(25, 27, 31);">3.3 写在最后：再谈 agent</font>
<font style="color:rgb(25, 27, 31);">事实上这种 LLM 主动使用工具并进行判断的方式并非 Self-RAG 首创，在此之前的</font><font style="color:rgb(25, 27, 31);"> </font>[AutoGPT](https://link.zhihu.com/?target=https%3A//github.com/Significant-Gravitas/AutoGPT)<font style="color:rgb(25, 27, 31);">,</font><font style="color:rgb(25, 27, 31);"> </font>[Toolformer](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2302.04761.pdf)<font style="color:rgb(25, 27, 31);">,</font><font style="color:rgb(25, 27, 31);"> </font>[ToolAlpaca](https://link.zhihu.com/?target=https%3A//github.com/tangqiaoyu/ToolAlpaca)<font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">和</font><font style="color:rgb(25, 27, 31);"> </font>[Graph-Toolformer](https://link.zhihu.com/?target=https%3A//www.linkresearcher.com/theses/88a5db6d-e0e6-41f2-9b1c-7838749b2432)<font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">中早已有之，而且支持多种 API 调用。</font>

<font style="color:rgb(25, 27, 31);">以</font><font style="color:rgb(25, 27, 31);"> </font>[Graph-Toolformer](https://link.zhihu.com/?target=https%3A//www.linkresearcher.com/theses/88a5db6d-e0e6-41f2-9b1c-7838749b2432)<font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">为例，其过程大致可分为以下几步：</font>

1. <font style="color:rgb(25, 27, 31);">针对 graph reasoning 任务设计少量 API Call 样本</font>
2. <font style="color:rgb(25, 27, 31);">基于 ChatGPT 对 prompt 进行 augmentation</font>
3. <font style="color:rgb(25, 27, 31);">使用现有 pre-train LLM 进行模型 fine-tuning</font>
4. <font style="color:rgb(25, 27, 31);">基于 external graph toolkits 的 graph reasoning</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292738085-fa6b7700-6703-4bbe-a13f-90c096bda103.webp)

<font style="color:rgb(145, 150, 161);">Graph-ToolFormer Framework</font>

<font style="color:rgb(25, 27, 31);">更近一步地，这些方式实际上都是以 LLM 为核心的，多 agent 协同模式下的具体呈现形式，在此范式下，LLM 充当思考和决策的中心，通过合理的 API 及工具的调用，完成各种复杂的任务和指令。</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292738465-c5914704-a7a6-4df5-b5d3-ac9cc5a85bc3.webp)

<font style="color:rgb(145, 150, 161);">LLM-powered autonomous agent system</font>

