<font style="color:rgba(0, 0, 0, 0.9);">在检索增强生成（Retrieval Augmented Generation：RAG）应用中实施检索步骤的常见做法是使用向量搜索。这种方法使用语义相似性查找相关段落。我们在 Azure Cognitive Search 中完全支持这种模式，并在矢量搜索的基础上提供了更多补充功能，从而显著提高了相关性。</font>

**<font style="color:rgba(0, 0, 0, 0.9);">Retrieval （检索）</font>****<font style="color:rgba(0, 0, 0, 0.9);"> </font>**<font style="color:rgba(0, 0, 0, 0.9);">--通常称为 L1，这一步骤的目标是从索引中快速找到满足搜索条件的所有文档--可能是数百万或数十亿的文档。对这些文档进行评分，选出前几名（通常在 50 名左右）返回给用户或提供给下一层。Azure 认知搜索支持 3 种不同的 L1 模式：</font>

+ <font style="color:rgba(0, 0, 0, 0.9);">Keyword：使用传统的全文搜索方法--通过特定语言的文本分析将内容分解为术语，创建反向索引以实现快速检索，并使用 BM25 概率模型进行评分。</font>
+ <font style="color:rgba(0, 0, 0, 0.9);">Vector：使用嵌入模型将文档从文本转换为矢量表示。检索是通过生成查询嵌入并找到其向量与查询向量最接近的文档来进行的。我们使用 Azure Open AI text-embedding-ada-002 (Ada-002) 嵌入和余弦相似性进行本帖中的所有测试。</font>
+ <font style="color:rgba(0, 0, 0, 0.9);">Hybrid：同时执行关键词和向量检索，并应用融合步骤从每种技术中选择最佳结果。Azure 认知搜索目前使用Reciprocal Rank Fusion （RRF）来生成单一结果集。</font>

<font style="color:rgba(0, 0, 0, 0.9);">  
</font>

**<font style="color:rgba(0, 0, 0, 0.9);">Ranking</font>****<font style="color:rgba(0, 0, 0, 0.9);"> </font>**<font style="color:rgba(0, 0, 0, 0.9);">--也称为 L2—使用top  L1 结果的子集，并计算更高质量的相关性分数来重新排列结果集。L2 可以提高 L1 的排名，因为它对每个结果应用了更强的计算能力。L2 排序器只能对 L1 已经找到的结果重新排序，如果 L1 错过了一个理想的文档，L2 就不能修复它。</font>

<font style="color:rgba(0, 0, 0, 0.9);">Semantic ranking由 Azure 认知搜索的 L2 ranker执行，该排序器利用了从微软必应改编而来的多语言深度学习模型。语义排名器可对 L1 结果中的前 50 个结果进行排名。</font>

<font style="color:rgba(0, 0, 0, 0.9);">截止到目前， 针对生成式AI ，</font>**<font style="color:rgba(0, 0, 0, 0.9);">Hybrid Retrieval + Semantic Ranking</font>**<font style="color:rgba(0, 0, 0, 0.9);">搜索的准确性是最高的。  
</font>

<font style="color:rgba(0, 0, 0, 0.9);">关键词检索和矢量检索从不同的角度处理检索问题，从而产生互补的能力。矢量检索从语义上将查询匹配到具有相似含义的段落。这一点非常强大，因为嵌入对拼写错误、同义词和措辞差异的敏感度较低，甚至可以在跨语言场景中发挥作用。关键词搜索非常有用，因为它可以优先匹配嵌入式中可能被淡化的特定重要词语。</font>

<font style="color:rgba(0, 0, 0, 0.9);">用户搜索可以有多种形式。在各种查询类型中，混合检索始终能将两种检索方法的优势发挥到极致。有了最有效的 L1，L2 排序步骤就能显著提高排在前列的结果的质量。</font>

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1702292823577-afbe5682-aa46-4ea6-a855-45720c9e9b3b.png)

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1702292823595-8a9259bf-e76e-41e6-87bf-582d17bfaf72.png)

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1702292823637-9ade25fd-e1ab-49c7-b761-d58d1099ec15.png)

<font style="color:rgba(0, 0, 0, 0.9);">  
</font>

<font style="color:rgba(0, 0, 0, 0.9);">接下来，我们展示一段代码。这段代码一共使用了几种搜索方法：</font>



1. <font style="color:rgba(0, 0, 0, 0.9);">纯向量搜索（Pure Vector Search）：只使用向量搜索一个字段。</font>
2. <font style="color:rgba(0, 0, 0, 0.9);">跨字段向量搜索（Cross-Field Vector Search）：在多个字段上进行向量搜索。</font>
3. <font style="color:rgba(0, 0, 0, 0.9);">多向量搜索（Multi-Vector Search）：使用多个向量进行搜索不同字段。</font>
4. <font style="color:rgba(0, 0, 0, 0.9);">带过滤器的纯向量搜索（Pure Vector Search with a Filter）：在进行纯向量搜索的同时，使用过滤器来限制搜索结果。它的目的是在保持向量搜索的高度灵活性和准确性的同时，通过过滤器来限制搜索结果的范围，从而提高搜索的效率和相关性。</font>
5. <font style="color:rgba(0, 0, 0, 0.9);">混合搜索（Hybrid Search）：同时使用文本搜索和向量搜索。</font>
6. **<font style="color:rgba(0, 0, 0, 0.9);">语义混合搜索（Semantic Hybrid Search）</font>**<font style="color:rgba(0, 0, 0, 0.9);">：在混合搜索的基础上，使用语义搜索来提取更加相关的结果。</font>

<font style="color:rgba(0, 0, 0, 0.9);">先查看做好的索引：</font>

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1702292823610-e09fe179-3d26-435c-8d22-d1b6e6086018.png)

<font style="color:rgba(0, 0, 0, 0.9);">以及语义搜索的设置：</font>

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1702292823692-7eaf4c2b-52b4-4fa1-bfe0-cc5b3a79c3e4.png)

<font style="color:rgba(0, 0, 0, 0.9);">由于代码较长，我们通过视频讲解的方式介绍这6种搜索。</font>

<font style="background-color:rgb(0, 0, 0);">，时长</font><font style="color:rgb(255, 255, 255);background-color:rgb(0, 0, 0);">21:24</font>

<font style="color:rgba(0, 0, 0, 0.9);">参考文档：</font>

<font style="color:rgba(0, 0, 0, 0.9);">https://techcommunity.microsoft.com/t5/azure-ai-services-blog/azure-cognitive-search-outperforming-vector-search-with-hybrid/ba-p/3929167</font>

```plain
https://github.com/Azure/cognitive-search-vector-pr/blob/main/demo-python/code/azure-search-vector-python-sample.ipynb
```

