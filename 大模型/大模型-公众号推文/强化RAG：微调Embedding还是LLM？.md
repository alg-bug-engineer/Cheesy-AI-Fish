

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1699520088237-d16233ba-d4ad-4ab8-be85-626c86342b27.png)

# 为什么我们需要微调？
微调有利于提高模型的效率和有效性。它可以减少训练时间和成本，因为它不需要从头开始。此外，微调可以通过利用预训练模型的功能和知识来提高性能和准确性。它还提供对原本无法访问的任务和领域的访问，因为它允许将预先训练的模型转移到新场景。换句话说，这一切都是为了获得更好的结果、减少奇怪的输出、更好地记住数据以及节省时间和金钱。

虽然微调也可用于使用外部数据“增强”模型，但微调可以通过多种方式补充RAG：

Embedding微调的好处

+ 微调Embedding模型可以在数据训练分布上实现更有意义的Embedding表示，从而带来更好的检索性能。

LLM微调的好处

+ 允许它学习给定数据集的风格
+ 允许它学习训练数据中可能较少出现的 DSL（例如 SQL）
+ 允许它纠正可能难以通过提示工程修复的幻觉/错误
+ 允许它将更好的模型（例如 GPT-4）提炼成更简单/更便宜的模型（例如 GPT-3.5、Llama 2）

简而言之，微调有助于更好的相似性搜索，这是获取正确数据以及生成正确答复所必须的前提。

微调主要有两种类型。第一个是微调Embedding，目的是提高数据检索的准确性，第二个是微调LLM，将领域知识注入到现有的LLM中。第一个是 RAG 特定的，而第二个是通用的。

# 微调Embedding
大型语言模型 (LLM) 可以处理广泛的任务，包括情感分析、信息提取和问答。正确的架构、深思熟虑的训练过程以及整个互联网训练数据的可用性结合在一起，使它们能够胜任某些任务。

LLM经过训练，可以使用这些大量数据在许多领域进行泛化，产生一个总体上很优秀但缺乏特定领域知识的模型。这时，微调就变得很重要。

微调过程涉及更改语言模型以更好地适应数据领域。例如，想要处理大量有关患者的医院文书工作，因此可能希望将LLM专门研究这些类型的文本。

LlamaIndex 关于微调Embedding的包含三个主要步骤：

1. 从数据生成综合问题-答案pari对数据集
2. 微调模型
3. 评估模型

## 微调Embedding
步骤总结：

1. 切分训练集和验证集
2. 使用LlamaIndex内置函数generate_qa_embedding_pairs生成训练数据集的问题/答案。此步骤将调用 LLM 模型（默认使用 OpenAI，可以替换为自己本地模型，例如ChatGLM、baichuan）来生成合成数据集
3. 使用SentenceTransformersFinetuneEngine与HuggingFace模型“m3e”模型进行微调。m3e模型可以提前下载到本地，避免网络访问的错误问题。

```plain
finetune_engine = SentenceTransformersFinetuneEngine( 
    train_dataset, 
    model_id= "path/to/m3e" , 
    model_output_path= "test_model" , 
    val_dataset=val_dataset, 
)
```

4. 使用hit rate 指标进行评估。

## 微调**<font style="color:rgb(36, 36, 36);">Adapter</font>**
这是微调Embedding的升级版本。基本的微调Embedding只需使用SentenceTransformersFinetuneEngine提供的开箱即用的功能即可。如果熟悉神经网络，那么layer、loss和 ReLU 等都不陌生。这个Adapter就是这样，让我们能够更好地控制微调过程。

步骤总结：

1. 与微调Embedding的步骤 1 类似，切分训练集和验证集
2. 类似于微调Embedding的步骤 2，构建合适的数据集
3. 使用 EmbeddingAdapterFinetuneEngine ，而不是使用 SentenceTransformersFinetuneEngine 。可以使用预定义的TwoLayerNN将图层作为参数添加到EmbeddingAdapterFinetuneEngine中，如下所示

```plain
base_embed_model = resolve_embed_model( "local:/path/to/m3e" ) 
adapter_model = TwoLayerNN( 
    384 ,   # 输入维度
    1024 ,   # 隐藏维度
    384 ,   # 输出维度
    bias= True , 
    add_residual= True , 
) 

finetune_engine = EmbeddingAdapterFinetuneEngine( 
    train_dataset, 
    base_embed_model, 
    model_output_path= "model5_output_test" , 
    model_checkpoint_path= "model5_ck" , 
    adapter_model=adapter_model, 
    epochs= 25 , 
    verbose= True , 
) 

## 如果需要从特定检查点加载模型，则使用
# load model from checkpoint in中间
embed_model_2layer_s900 = AdapterEmbeddingModel( 
    base_embed_model, 
    "model5_ck/step_900" , 
    TwoLayerNN, 
)
```

4. 与基本微调Embedding的步骤4类似

## <font style="color:rgb(36, 36, 36);">Router</font>微调
我自己并不经常使用这种微调，这种类型的微调对于router查询很有用。但路由器查询是非常特定于数据域的，添加这种Embedding只会增加 RAG 的复杂性。

路由器的快速总结：不能扔一堆文档进行Embedding，然后在其上构建检索。这种方法不会给你带来任何好的结果，甚至是一个不可接受的结果。因此 LlamaIndex 引入了一个奇妙的概念，称为 Router。路由器是 LLM 实现自动化决策的重要一步，这本质上将 LLM 视为分类器

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1699522005241-636384a3-fdd6-40ac-b906-5977800b74a7.png)

但是，基础路由器有时很差劲，查询和索引之间的匹配率非常低。为了解决这个问题，LlamaIndex 现在可以微调路由器。这将有助于减少每个查询运行的循环数量，因此期望结果更快。但结果有时还是很可怕。

基本上，对于每个文档，在其上构建多个索引，例如 VectorIndex、SummaryIndex、KeywordIndex 等，然后给出每个索引的元数据或描述，然后在此基础上构建代理，并使用元数据描述来告诉 LLM这个代理是做什么的。如果有 100 万份文档，那么就有 100 万个代理。每次进行查询时，LLM 都需要通过 100 万个代理来找出最适合用来回答问题的代理。因此，它是非常慢的。为了解决这个问题，LlamaIndex 将当前版本升级到另一个版本，该版本基本上是在文档（工具）检索期间重新排名代理可以用来规划的查询规划工具。

仅当设计的 RAG 系统以路由器为中心时，否则，ReAct 代理或多代理是更好的方法。

## <font style="color:rgb(36, 36, 36);">Cross-Encoder</font>微调
简而言之，Bi-Encoder 就是使用双编码器，将句子 A 和句子 B 转换为句子Embedding A1 和句子Embedding B1。然后可以使用余弦相似度来比较这些句子Embedding。

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1699520087815-1e0f57d0-d064-4ae0-be46-84d5092e3c72.png)

相反，对于交叉编码器，我们将两个句子同时传递到 Transformer 网络。它产生一个介于 0 和 1 之间的输出值，表示输入句子对的相似度

交叉编码器不会产生句子Embedding。此外，我们无法将单个句子传递给交叉编码器。

交叉编码器比双编码器具有更好的性能。然而，对于许多应用来说，它们并不实用，因为它们不产生Embedding，我们可以使用余弦相似度进行索引或有效比较。

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1699520087919-b0fd90d0-3c5b-4578-8384-99729313b16b.png)

交叉编码器比双编码器具有更高的性能，但是，它们对于大型数据集的扩展性不佳。在这里，结合交叉编码器和双向编码器是有意义的，例如在信息检索/语义搜索场景中：首先，使用高效的双向编码器来检索查询的前 100 个最相似的句子。然后，使用交叉编码器通过计算每个（查询、命中）组合的分数来重新排名这 100 个命中。

# 微调LLM
因此，已经完成了Embedding的微调，如上所述，微调Embedding有助于提高数据检索的准确性。如果是这样，我们是否需要对LLM进行微调？

因为并非每次都需要 RAG。开发功能齐全的 RAG 每一步都很复杂。拥有一个好的RAG应用程序需要一个由优秀的软件工程师组成的团队来开发前端和可扩展的后端，优秀的数据工程师来处理用于开发RAG的数据管道和多个数据库，一些优秀的机器学习工程师+数据科学家开发模型并对文本块、Embedding性能、良好的数据检索方法进行实验，然后合成数据、路由器、代理等。更不用说将需要良好的 MLOps 来监控 RAG 的性能。

如果可以通过在新数据上逐步微调 LLM 来简化所有这些方法，会怎么样？使其成为 ChatGPT，但根据自己的数据进行微调。会更容易吗？

大多数LLM/RAG以PoC为主。它可以处理小数据集并在非常特定的情况下处理得很好，但很难扩展或处理现实生活中的用例。

但我们假设有资金定期调整LLM课程。我们该怎么做呢？

LlamaIndex 有多种选项可以帮助微调的LLM。主要目的是改进较小模型以超越较大参数规模模型。假设 GPT-4 对你的应用程序来说非常好，但它会让公司破产，因为它很昂贵。GPT-3.5 更便宜，性能也可以接受，但希望 GPT-4 的性能让的客户满意。那么你可能会想到微调LLM。

## 为什么要微调LLM
如前所述，微调不仅可以提高基本模型的性能，而且较小（微调）的模型通常可以在训练它的任务集上胜过较大（更昂贵）的模型。OpenAI 通过其第一代“InstructGPT”模型证明了这一点，其中 1.3B 参数 InstructGPT 模型补全优于 175B 参数 GPT-3 基本模型，尽管其尺寸要小 100 倍。

其中一大问题是LLM的背景知识是有限的。因此，该模型可能在需要大量知识库或特定领域信息的任务上表现不佳。微调模型可以通过在微调过程中“学习”这些信息，或者换句话说，使用最新数据更新模型来避免此问题。GPT-4 仅拥有 2023年3月之前的知识。微调 LLM 将使用的私人数据更新模型并减少幻觉，也不需要 RAG，因为微调 LLM 已经更新了的数据。



改进 RAG 很困难，有多个步骤，根据我的经验，可以显着改进 RAG 的最重要步骤是文本块和Embedding。因此，微调Embedding模型是必要的（但是不是必须的）步骤。此外，微调LLM将更新现有LLM的行为，从而减少响应中的幻觉并提供更好的综合答案。

