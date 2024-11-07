## [大模型](https://www.zhihu.com/search?q=%E5%A4%A7%E6%A8%A1%E5%9E%8B&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">的“幻觉”问题</font>
<font style="color:rgb(25, 27, 31);">从ChatGPT出现使得</font>[大语言模型](https://www.zhihu.com/search?q=%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">( Large Language Model, LLM ) 在各个领域应用爆火起来。目前LLM出现的幻觉问题仍旧是一个重要挑战。</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292884806-8e22cdc7-3aee-4d3c-883d-3dd61e0a5a2f.webp)

<font style="color:rgb(145, 150, 161);">RAG新时代</font>

## <font style="color:rgb(25, 27, 31);">为什么会出现幻觉问题</font>
<font style="color:rgb(25, 27, 31);">大型语言模型是经过训练的机器学习模型，它根据我们提供的提示</font>[生成文本](https://www.zhihu.com/search?q=%E7%94%9F%E6%88%90%E6%96%87%E6%9C%AC&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">。这种模型最大的特点就是从训练数据中获取知识，但是又很难判断模型记住了哪些知识，没有记住哪些知识。这就导致，在模型生成文本的时候，这个模型无法判断生成的是否正确。</font>

<font style="color:rgb(83, 88, 97);">“幻觉”是指模型生成不正确、无意义或不真实文本的现象。因为LLM不是</font>[数据库](https://www.zhihu.com/search?q=%E6%95%B0%E6%8D%AE%E5%BA%93&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(83, 88, 97);">或者</font>[搜索引擎](https://www.zhihu.com/search?q=%E6%90%9C%E7%B4%A2%E5%BC%95%E6%93%8E&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(83, 88, 97);">，并不会知道他们引用的回答基于何处。生成的结果不一定是基于训练数据学习来的， 而是依靠提升生成最相关的。[1]</font>

**<font style="color:rgb(25, 27, 31);">幻觉的基本定义</font>**<font style="color:rgb(25, 27, 31);">：大模型生成看似合理的内容，其实这些内容是不正确的或者是与输入Prompt无关，甚至是有冲突的现象。</font>

<font style="color:rgb(83, 88, 97);">来自人工智能研究中心（Center for Artificial Intelligence Research ）的一篇研究论文[2]将 LLM 的幻觉定义为“生成的内容与提供的源内容不符或没有意义”。</font>

<font style="color:rgb(25, 27, 31);">要理解幻觉，可以从一些文本中构建两个字母的二元</font>[马尔可夫模型](https://www.zhihu.com/search?q=%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E6%A8%A1%E5%9E%8B&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">：提取一长段文本，构建每对相邻字母的表格并计算计数。例如，“大语言模型中的幻觉”会产生“HA”、“AL”、“LL”、“LU”等，并且有一次计数“LU”和两次计数“LA”。现在，如果您以“L”提示符开始，您产生“LA”的可能性是“LL”或“LS”的两倍。然后，在提示“LA”的情况下，您有相同的概率说出“AL”、“AT”、“AR”或“AN”。然后您可以尝试使用“LAT”提示并继续此过程。最终，这个模型发明了一个不存在的新词。这是</font>[模式统计](https://www.zhihu.com/search?q=%E6%A8%A1%E5%BC%8F%E7%BB%9F%E8%AE%A1&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">的结果，放在大LLVM上就是大模型幻觉的一种例子。</font>

### <font style="color:rgb(25, 27, 31);">幻觉的分类</font>
<font style="color:rgb(25, 27, 31);">幻觉可以分为几种类型：</font>

+ [逻辑谬误](https://www.zhihu.com/search?q=%E9%80%BB%E8%BE%91%E8%B0%AC%E8%AF%AF&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">：模型在进行推理时出现了错误，提供错误的答案。</font>
+ **<font style="color:rgb(25, 27, 31);">捏造事实</font>**<font style="color:rgb(25, 27, 31);">：模型自信地断言不存在的事实，而不是回答“我不知道”。例如：谷歌的 AI 聊天机器人 Bard 在第一次公开演示中犯了一个事实错误[3]。</font>
+ **<font style="color:rgb(25, 27, 31);">数据驱动的偏见</font>**<font style="color:rgb(25, 27, 31);">：由于某些数据的普遍存在，模型的输出可能会偏向某些方向，导致错误的结果。例如：</font>[自然语言处理](https://www.zhihu.com/search?q=%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">模型中发现的政治偏见[4]。</font>

## <font style="color:rgb(25, 27, 31);">什么是检索增强LLM</font>
<font style="color:rgb(25, 27, 31);">前面提到了LLM存在的一个严重的问题就是</font>[幻觉问题](https://www.zhihu.com/search?q=%E5%B9%BB%E8%A7%89%E9%97%AE%E9%A2%98&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">，另外一个问题的就是LLM对于实效性较强的问题难以做出回答过着给出过时回答，为了解决这两种问题，试图通过检索外部相关信息的方式来提升LLM的生成能力，这就称之为检索增强生成（Retrieval-augmented Generation，RAG）。</font>

**<font style="color:rgb(25, 27, 31);">RAG可从外部知识库检索事实，以最准确、最新的信息为基础的大语言模型 (LLM)，并让用户深入了解 LLM 的生成过程。它确保模型能够访问最新、可靠的事实，并且用户能够访问模型的来源，确保可以检查其声明的准确性和准确性。</font>**

<font style="color:rgb(25, 27, 31);">目前很多AI团队都会有效考虑RAG的方式，并且基于这个最佳实践引发了大量工具，比如</font>[向量数据库](https://www.zhihu.com/search?q=%E5%90%91%E9%87%8F%E6%95%B0%E6%8D%AE%E5%BA%93&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">。</font>

### <font style="color:rgb(25, 27, 31);">RAG的特点</font>
<font style="color:rgb(25, 27, 31);">说起来RAG的特点可以总结以下几条：</font>

1. <font style="color:rgb(25, 27, 31);">RAG 是一种相对较新的</font>[人工智能技术](https://www.zhihu.com/search?q=%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E6%8A%80%E6%9C%AF&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">，可以通过允许大型语言模型 (LLM) 在无需重新训练的情况下利用额外的数据资源来提高生成式 AI 的质量。</font>
2. <font style="color:rgb(25, 27, 31);">RAG 模型基于组织自身的数据构建知识存储库，并且存储库可以不断更新，以帮助生成式 AI 提供及时的上下文答案。</font>
3. <font style="color:rgb(25, 27, 31);">使用自然语言处理的聊天机器人和其他对话系统可以从 RAG 和</font>[生成式人工智能](https://www.zhihu.com/search?q=%E7%94%9F%E6%88%90%E5%BC%8F%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">中受益匪浅。</font>
4. <font style="color:rgb(25, 27, 31);">实施 RAG 需要</font>[矢量数据库](https://www.zhihu.com/search?q=%E7%9F%A2%E9%87%8F%E6%95%B0%E6%8D%AE%E5%BA%93&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">等技术，这些技术可以快速编码新数据，并搜索该数据以输入给LLM模型。</font>

### <font style="color:rgb(25, 27, 31);">RAG的两个阶段</font>
<font style="color:rgb(25, 27, 31);">RAG在整体划分为</font>**<font style="color:rgb(25, 27, 31);">检索和生成</font>**<font style="color:rgb(25, 27, 31);">两个阶段：</font>

**<font style="color:rgb(25, 27, 31);">检索阶段</font>**<font style="color:rgb(25, 27, 31);">：在检索阶段，算法搜索并检索与用户提示或问题相关的信息片段。下图中的Step 1 和 Step 2从想向量数据库中查找与Query相关的数据。</font>

    - <font style="color:rgb(25, 27, 31);">开放域的消费者环境中，这些事实可以来自互联网上的索引文档；</font>
    - <font style="color:rgb(25, 27, 31);">在封闭域的企业环境中，通常使用较小的一组源来提高安全性和可靠性。</font>

**<font style="color:rgb(25, 27, 31);">生成阶段</font>**<font style="color:rgb(25, 27, 31);">：大模型从增强提示及其训练数据的内部表示中提取信息，以在那一刻为用户量身定制引人入胜的答案。然后可以将答案传递给</font>[聊天机器人](https://www.zhihu.com/search?q=%E8%81%8A%E5%A4%A9%E6%9C%BA%E5%99%A8%E4%BA%BA&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">，并附上其来源的链接。</font>

<font style="color:rgb(25, 27, 31);">  
</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292884839-c7aa4444-f0aa-4d96-abaf-ecb52241cf1b.webp)

<font style="color:rgb(145, 150, 161);">RAG基本处理流程</font>

### <font style="color:rgb(25, 27, 31);">RAG面临的一些挑战</font>
<font style="color:rgb(25, 27, 31);">RAG本身就就是一项相对较新的技术，在2020年首次提出，目前绝大部分场景仍处于探索过程中，就目前阶段而言，不得不的面临的一些挑战是：</font>

+ **<font style="color:rgb(25, 27, 31);">技术太新及其研究较少</font>**
+ **<font style="color:rgb(25, 27, 31);">如何对知识库和向量数据库中</font>**[结构化](https://www.zhihu.com/search?q=%E7%BB%93%E6%9E%84%E5%8C%96&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)**<font style="color:rgb(25, 27, 31);">和</font>**[非结构化数据](https://www.zhihu.com/search?q=%E9%9D%9E%E7%BB%93%E6%9E%84%E5%8C%96%E6%95%B0%E6%8D%AE&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)**<font style="color:rgb(25, 27, 31);">建模</font>**
+ **<font style="color:rgb(25, 27, 31);">整个RAG</font>**[工程化模型](https://www.zhihu.com/search?q=%E5%B7%A5%E7%A8%8B%E5%8C%96%E6%A8%A1%E5%9E%8B&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)**<font style="color:rgb(25, 27, 31);">面临的流程问题</font>**
+ **<font style="color:rgb(25, 27, 31);">如何处理不准确信息来源以及如何剔除不准确信息</font>**

---

<font style="color:rgb(25, 27, 31);">ChatGPT肯定听说过吧，他在微软的Bing上面的应用就是RAG的场景，其实这个RAG本质上就是搜索与LLM的结合，</font>[搜索技术](https://www.zhihu.com/search?q=%E6%90%9C%E7%B4%A2%E6%8A%80%E6%9C%AF&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">从发展到现在已经十几年时间了，而LLM技术才不过区区2、3年，如果想要看看这当下AI模型发展的新时代，还是需要有个指路人。这不，「知乎</font>[知学堂](https://www.zhihu.com/search?q=%E7%9F%A5%E5%AD%A6%E5%A0%82&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">」推出了为期两天《AGI大模型进阶之旅》的公开课，</font>**<font style="color:rgb(25, 27, 31);">不仅让你倾听行业顶尖大佬来揭秘未来AI发展潮流， 还能与行业大佬面对面交流，领取行业大佬提供的学习资料！</font>**

**<font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">2023超</font>****<font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">🔥</font>****<font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">的AI大模型公开课</font>****<font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">👉</font>****<font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">大模型资料包免费领！</font>**

**<font style="color:rgb(255, 80, 26);background-color:rgb(248, 248, 250);">￥0.00</font>****<font style="color:rgb(255, 80, 26);background-color:rgb(248, 248, 250);">立即体验</font>**

<font style="color:rgb(25, 27, 31);">如果还是0元，建议你赶紧冲，跟AI大牛对话的机会太难得了！除此以外，更能体验自主训练的机器学习模型，实践理论相结合。</font>**<font style="color:rgb(25, 27, 31);">上面的链接就是公开课的链接！！另外，添加课程之后一定一定一定要添加助教小姐姐的微信，可以私聊助教领取今年最火最热的大模型学习资源！！</font>**

---

## <font style="color:rgb(25, 27, 31);">RAG来解决的问题</font>
### **<font style="color:rgb(25, 27, 31);">长尾问题</font>**
<font style="color:rgb(25, 27, 31);">目前的LLM模型训练数据非常庞大，参数量也非常多，训练数据来源丰富。在有限的参数上学习无穷的知识、理解无穷的信息是不现实的。这就导致对于经常出现和相对大众化的知识，LLM通常能够得出比较正确的结果，而对于一些</font>[长尾知识](https://www.zhihu.com/search?q=%E9%95%BF%E5%B0%BE%E7%9F%A5%E8%AF%86&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">，通常回复并不可靠。其中ICML会议上的</font><font style="color:rgb(25, 27, 31);"> </font>[Large Language Models Struggle to Learn Long-Tail Knowledge](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2211.08411)<font style="color:rgb(25, 27, 31);">[3]，就研究了 LLM 对基于事实的问答的准确性和</font>[预训练数据](https://www.zhihu.com/search?q=%E9%A2%84%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">中相关领域文档数量的关系，发现有很强的相关性，即预训练数据中相关文档数量越多，LLM 对事实性问答的回复准确性就越高，并且对长尾知识学习能力较弱。</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292884772-9a174c53-ad7c-419b-b5b9-c38c11eb916b.webp)

<font style="color:rgb(145, 150, 161);">LLM的长尾问题</font>

<font style="color:rgb(25, 27, 31);">为了增加LLM对于长尾知识的学习能力，很容易就是增加更多相关的长尾知识，确实这么做提升了对长尾知识的预测能力。</font>

### <font style="color:rgb(25, 27, 31);">私有数据</font>
<font style="color:rgb(25, 27, 31);">像OpenAI这种模型大多数都是使用公开数据进行学习，但是如果真想把数据应用到某个公司或者某个公司的内部相关知识，就必须学习特定的私有数据进行学习。但是，如果训练数据中包含了某个公司的私有信息，也存在一个问题，就是隐私信息泄漏的问题。就比如当初Bing出现的时候出现的Windows的密钥出来的情况，如果暴露的是公司</font>[内部信息](https://www.zhihu.com/search?q=%E5%86%85%E9%83%A8%E4%BF%A1%E6%81%AF&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">更严重。</font>

**<font style="color:rgb(25, 27, 31);">目前的最佳实践是，将私有数据作为一个</font>**[外部数据库](https://www.zhihu.com/search?q=%E5%A4%96%E9%83%A8%E6%95%B0%E6%8D%AE%E5%BA%93&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)**<font style="color:rgb(25, 27, 31);">，让LLM在回答私有数据问题时候，直接从外部数据检索相关信息，在结合检索出来的内容进行回答。这样不参与训练，就不会在LLM的模型参数中，记住私有知识。</font>**

### <font style="color:rgb(25, 27, 31);">信息更新</font>
<font style="color:rgb(25, 27, 31);">OpenAI推出的ChatGPT模型现在都知道模型用的数据信息截止在2021年底，因为训练时间长、数据量大，所以一般而言不会重新数据。因此就会导致LLM从原来的历史数据中没有办法覆盖到2021年以后的数据，这样的实效性问题就没办法解决。</font>

<font style="color:rgb(25, 27, 31);">RAG就可以把更新的知识放在外部数据库，在问题的时候检索最新的知识，并经过LLM上进行更新和拓展，解决新鲜度问题。</font>

### <font style="color:rgb(25, 27, 31);">可解释性</font>
<font style="color:rgb(25, 27, 31);">AI发展到至今，一直在考虑研究的一个问题就是可解释性。尤其是在端到端的训练过程，</font>[神经网络模型](https://www.zhihu.com/search?q=%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%A8%A1%E5%9E%8B&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">逐渐变成黑盒。LLM也不例外，一般而言LLM的输出没有信息来源，这也就很难解释为什么这么生成回答。通过RAG可以解决</font>**<font style="color:rgb(25, 27, 31);">可解释性、信息溯源、信息验证</font>**<font style="color:rgb(25, 27, 31);">等问题，一旦检索的内容和生辰的内容建立的关系就可以知道LLM模型根据哪些信息得出的回答。</font>

<font style="color:rgb(25, 27, 31);">当初的Bing的就是这么落地的，不仅仅提供信息生成内容，并且提供了信息来源。</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292884803-a343c992-6480-49e9-a29f-35ee79ed1e40.webp)

<font style="color:rgb(145, 150, 161);">Bing的可解释性回答</font>

## <font style="color:rgb(25, 27, 31);">RAG工程化的优化方案</font>
<font style="color:rgb(25, 27, 31);">部署有效的 RAG 系统需要进行大量实验来优化每个组件，包括</font>[数据收集](https://www.zhihu.com/search?q=%E6%95%B0%E6%8D%AE%E6%94%B6%E9%9B%86&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">、模型嵌入、</font>[分块策略](https://www.zhihu.com/search?q=%E5%88%86%E5%9D%97%E7%AD%96%E7%95%A5&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">等。目前还没有一个</font>[放之四海而皆准](https://www.zhihu.com/search?q=%E6%94%BE%E4%B9%8B%E5%9B%9B%E6%B5%B7%E8%80%8C%E7%9A%86%E5%87%86&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">的准则。部署有原则的 RAG 设计需要进行全面的实验，通常涉及数据收集、模型嵌入、分块策略等的迭代。</font>

<font style="color:rgb(25, 27, 31);">  
</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292884819-ab6a24c8-4e37-4a36-8da0-27b2d7db764c.webp)

<font style="color:rgb(145, 150, 161);">RAG工程化实践方案</font>

<font style="color:rgb(25, 27, 31);">  
</font>

<font style="color:rgb(25, 27, 31);">按照步骤可以将其分为</font>**<font style="color:rgb(25, 27, 31);">数据索引模块、</font>**[数据检索模块](https://www.zhihu.com/search?q=%E6%95%B0%E6%8D%AE%E6%A3%80%E7%B4%A2%E6%A8%A1%E5%9D%97&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)**<font style="color:rgb(25, 27, 31);">以及LLM生成模块</font>**<font style="color:rgb(25, 27, 31);">。这里仅仅讲一些评估方法以及需要注意的点，毕竟新技术仍处于快速发展研究状态，没有一个完整确切的解决方案。</font>

### [数据索引](https://www.zhihu.com/search?q=%E6%95%B0%E6%8D%AE%E7%B4%A2%E5%BC%95&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">模块</font>
<font style="color:rgb(25, 27, 31);">与监督学习不同，检索增强生成（RAG）系统的性能高度依赖于它们所训练的数据的质量。RAG 系统的工作原理是从知识库中检索相关信息，然后使用该信息生成响应。</font>**<font style="color:rgb(25, 27, 31);">如果知识库中的数据质量较差，RAG 将无法生成准确或信息丰富的结果。</font>**

<font style="color:rgb(25, 27, 31);">有几个关键因素会影响 RAG 系统的数据：</font>

+ **<font style="color:rgb(25, 27, 31);">数据质量</font>**<font style="color:rgb(25, 27, 31);">：如果数据不准确、不完整或有偏差，RAG 系统将更有可能生成不准确或误导性的响应。</font>
+ **<font style="color:rgb(25, 27, 31);">数据准备</font>**<font style="color:rgb(25, 27, 31);">：这包括清理数据、删除重复条目以及将数据转换为与 RAG 系统兼容的格式。</font>
+ **<font style="color:rgb(25, 27, 31);">数据源</font>**<font style="color:rgb(25, 27, 31);">：如果知识库仅包含来自有限数量的源的数据，则 RAG 系统可能无法生成像可以访问更广泛的数据源那样全面或信息丰富的响应。</font>
+ [元数据](https://www.zhihu.com/search?q=%E5%85%83%E6%95%B0%E6%8D%AE&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">：通过提供有关上下文中传递的块的元数据，法学硕士将能够更好地理解上下文，从而可能产生更好的输出。</font>
+ **<font style="color:rgb(25, 27, 31);">附加上下文和知识</font>**<font style="color:rgb(25, 27, 31);">：如果可以的话，使用</font>[知识图谱](https://www.zhihu.com/search?q=%E7%9F%A5%E8%AF%86%E5%9B%BE%E8%B0%B1&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">可以通过在查询时提供附加上下文来增强 RAG 应用程序，从而使系统能够生成更准确、信息更丰富的响应。</font>

### <font style="color:rgb(25, 27, 31);">数据分块</font>
<font style="color:rgb(25, 27, 31);">在RAG中，“分块”是指将输入的长文本分割成简洁、有意义的单元，这是因为LLM对于上下文的长度是有限制的，不能接受长文本信息，另外还会增加无效的额外信息进行干扰。</font>

**<font style="color:rgb(25, 27, 31);">有效分块能够促进</font>**[检索系统](https://www.zhihu.com/search?q=%E6%A3%80%E7%B4%A2%E7%B3%BB%E7%BB%9F&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)**<font style="color:rgb(25, 27, 31);">精确定位相关上下文段落以生成响应。这些块的质量和结构对于系统的功效至关重要，确保检索到的文本针对用户的查询进行精确定制。</font>**

<font style="color:rgb(25, 27, 31);">这里的分块实验数据可以借鉴Anyscale[4]的实验结论，</font>

+ <font style="color:rgb(25, 27, 31);">他们发现分块显著影响生成内容质量</font>
+ <font style="color:rgb(25, 27, 31);">较大的分块窗口效果明显，到那时超过某个最佳窗口后就开始减弱</font>
+ <font style="color:rgb(25, 27, 31);">虽然较大的块大小可以提高性能，但过多的上下文可能会引入噪音</font>

### <font style="color:rgb(25, 27, 31);">嵌入模型</font>
<font style="color:rgb(25, 27, 31);">这个部分是如何将数据向量化的过程，促进信息检索的</font>[语义搜索](https://www.zhihu.com/search?q=%E8%AF%AD%E4%B9%89%E6%90%9C%E7%B4%A2&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">。同样的Anyscale的实验证明：</font>

<font style="color:rgb(25, 27, 31);">嵌入模型的选择会显着影响检索和质量得分，对于特定任务，较小的模型甚至优于排名最高的模型。</font>**<font style="color:rgb(25, 27, 31);">事实证明，简单地从排行榜中选择表现最好的</font>**[嵌入模型](https://www.zhihu.com/search?q=%E5%B5%8C%E5%85%A5%E6%A8%A1%E5%9E%8B&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)**<font style="color:rgb(25, 27, 31);">并不总是最好的决定！</font>**

<font style="color:rgb(25, 27, 31);">嵌入模型和分块策略都会显着影响 RAG 系统的性能。然而，</font>**<font style="color:rgb(25, 27, 31);">分块策略的影响似乎稍大一些</font>**<font style="color:rgb(25, 27, 31);">。嵌入的价值很大程度上取决于您的用例。现成的嵌入模型可以为数据块生成适合大多数用例的嵌入。如果正在研究特定领域，这些模型可能无法充分表示</font>[向量空间](https://www.zhihu.com/search?q=%E5%90%91%E9%87%8F%E7%A9%BA%E9%97%B4&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">中的领域，从而导致检索质量较差。</font>

### <font style="color:rgb(25, 27, 31);">数据检索模块</font>
<font style="color:rgb(25, 27, 31);">查询文本的表达方法直接影响着检索结果，微小的文本改动都可能会得到天差万别的结果。许多向量数据库允许您使用混合（基于规则和向量搜索）检索方法。考虑到语言模型的限制以及领域专业知识在排名结果中的关键作用，根据特定需求定制</font>[检索方法](https://www.zhihu.com/search?q=%E6%A3%80%E7%B4%A2%E6%96%B9%E6%B3%95&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">变得至关重要。</font>

<font style="color:rgb(25, 27, 31);">目前绝大部分采用的是</font>[向量搜索](https://www.zhihu.com/search?q=%E5%90%91%E9%87%8F%E6%90%9C%E7%B4%A2&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">的方式来进行查询，其中处理查询的方式也有很多，这里就不做赘述，不过查询后面的一步又为重要，就是</font>**<font style="color:rgb(25, 27, 31);">排序和后处理</font>**<font style="color:rgb(25, 27, 31);">。</font>

<font style="color:rgb(25, 27, 31);">经过前面的检索过程可能会得到很多相关文档，就需要进行筛选和排序。常用的筛选和排序策略包括：</font>

+ <font style="color:rgb(25, 27, 31);">基于</font>[相似度分数](https://www.zhihu.com/search?q=%E7%9B%B8%E4%BC%BC%E5%BA%A6%E5%88%86%E6%95%B0&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">进行过滤和排序</font>
+ <font style="color:rgb(25, 27, 31);">基于关键词进行过滤，比如限定包含或者不包含某些关键词</font>
+ <font style="color:rgb(25, 27, 31);">让 LLM 基于返回的相关文档及其相关性得分来重新排序</font>
+ <font style="color:rgb(25, 27, 31);">基于时间进行过滤和排序，比如只筛选最新的相关文档</font>
+ <font style="color:rgb(25, 27, 31);">基于时间对相似度进行</font>[加权](https://www.zhihu.com/search?q=%E5%8A%A0%E6%9D%83&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">，然后进行排序和筛选</font>

<font style="color:rgb(25, 27, 31);">  
</font>

![](https://cdn.nlark.com/yuque/0/2023/webp/406504/1702292885195-9699d335-3043-455a-8928-5a253d4dc245.webp)

<font style="color:rgb(145, 150, 161);">数据检索</font>

<font style="color:rgb(25, 27, 31);">  
</font>

### <font style="color:rgb(25, 27, 31);">LLM生成模块</font>
<font style="color:rgb(25, 27, 31);">LLM通过利用相关文档片段中的信息生成了精确的答案。LLM通过外部数据得到增强，提高了其响应质量。这个部分的生成策略可以简单划分为三种：</font>

+ <font style="color:rgb(25, 27, 31);">同一个</font>[LLM模型](https://www.zhihu.com/search?q=LLM%E6%A8%A1%E5%9E%8B&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">依次结合每一个检索出来的相关文本，并修正生成</font>
+ <font style="color:rgb(25, 27, 31);">多个LLM模型针对检索出来的所有相关文本进行生成，并策略生成</font>
+ <font style="color:rgb(25, 27, 31);">上面两种的</font>[混合模型](https://www.zhihu.com/search?q=%E6%B7%B7%E5%90%88%E6%A8%A1%E5%9E%8B&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)

## <font style="color:rgb(25, 27, 31);">结论</font>
<font style="color:rgb(25, 27, 31);">这里简单总结一下，RAG的最佳实践方式并没有一个通用完整的方案，不过针对工业落地有着一个可以参考的调优步骤：</font>

+ <font style="color:rgb(25, 27, 31);">分组调优：分别保证每个模块在相应条件下达到最优</font>
+ [组合调优](https://www.zhihu.com/search?q=%E7%BB%84%E5%90%88%E8%B0%83%E4%BC%98&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">：在前面调优完成后整体判断模型性能</font>
+ <font style="color:rgb(25, 27, 31);">建立有效的指标： 比如</font>**<font style="color:rgb(25, 27, 31);">Retrieval_Score</font>**<font style="color:rgb(25, 27, 31);">和</font>**<font style="color:rgb(25, 27, 31);">Quality_Score</font>**<font style="color:rgb(25, 27, 31);">来分析查询嵌入和</font>[知识库](https://www.zhihu.com/search?q=%E7%9F%A5%E8%AF%86%E5%BA%93&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3309968693%7D)<font style="color:rgb(25, 27, 31);">嵌入块之间的距离来衡量检索到的上下文的质量</font>

