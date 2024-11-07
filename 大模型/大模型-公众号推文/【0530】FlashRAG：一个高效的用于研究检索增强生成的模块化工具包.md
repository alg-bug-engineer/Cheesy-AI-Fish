<font style="color:rgb(88, 90, 115);">本文介绍了一种名为FlashRAG的工具包，旨在为研究人员提供一个统一的框架，以实现和比较各种基于检索增强生成（RAG）技术的研究方法。该工具包包括了12个先进的RAG方法，并收集整理了32个基准数据集。此外，它还具有可定制化的模块化框架、丰富的预实现RAG工作集合、全面的数据集、高效的辅助预处理脚本以及广泛的标准评估指标等特点。FlashRAG工具包及其资源可在GitHub上获得。</font>

## <font style="color:rgb(88, 90, 115);">1. 引言</font>


<font style="color:rgb(88, 90, 115);">随着大语言模型（LLMs）的发展，检索增强生成（RAG）技术越来越受到研究者的关注。RAG通过利用外部知识库来缓解LLM中的幻觉问题[1,2]。然而，由于缺乏标准化的实现框架，加上RAG过程本身的复杂性，研究人员很难在一致的环境中比较和评估不同的RAG方法。尽管已有一些RAG工具包如LangChain和LlamaIndex，但它们通常比较重量级且难以满足研究人员的个性化需求。</font>



<font style="color:rgb(88, 90, 115);">为此，我们提出了FlashRAG，一个高效且模块化的开源工具包，旨在帮助研究人员在统一的框架下复现已有的RAG方法并开发自己的RAG算法。我们的工具包实现了12个先进的RAG方法，并收集整理了32个基准数据集。</font>



## <font style="color:rgb(88, 90, 115);">2. FlashRAG总体架构</font>


<font style="color:rgb(88, 90, 115);">如图1所示，FlashRAG由三个模块组成：环境模块、组件模块和管道模块。</font>

![](https://cdn.nlark.com/yuque/0/2024/png/406504/1716944504176-d894a3c8-4959-4c5a-aded-fa93323e110e.png)

![](https://github.com/RUC-NLPIR/FlashRAG/raw/main/docs/assets/overview.png)

<font style="color:rgb(88, 90, 115);">FlashRAG是一个用于支持相关性问答（RAG）研究的工具包。它由三个模块组成：环境模块、组件模块和管道模块。其中，环境模块为实验提供了必要的数据集、超参数和评估指标；组件模块包括五个主要组件：判断器、检索器、排序器、精炼器和生成器；管道模块将各种组件组合在一起以实现完整的RAG过程。</font>

<font style="color:rgb(88, 90, 115);">在组件模块中，有五种主要组件可供选择：</font>

1. <font style="color:rgb(88, 90, 115);">判断器（Judger）：根据SKR方法确定是否需要检索。</font>
2. <font style="color:rgb(88, 90, 115);">检索器（Retriever）：支持多种检索方式，如稀疏检索和稠密检索，并且支持自定义预取结果。</font>
3. <font style="color:rgb(88, 90, 115);">排序器（Reranker）：用于优化检索结果的顺序，支持多种跨编码模型。</font>
4. <font style="color:rgb(88, 90, 115);">精炼器（Refiner）：用于减少生成器的输入文本量并降低噪声，提供四种类型的精炼器供选择。</font>
5. <font style="color:rgb(88, 90, 115);">生成器（Generator）：支持多种主流LLM加速库和编码解码模型。</font>

<font style="color:rgb(88, 90, 115);">在管道模块中，用户可以根据自己的需求组装所需的组件，以便执行整个RAG过程。同时，该工具包还提供了多个预设的管道，方便用户快速搭建RAG流程。</font>



<font style="color:rgb(88, 90, 115);">FlashRAG相较于其他RAG工具包的优势在于其丰富的组件和可扩展性。用户可以根据具体任务的需求选择合适的组件，并通过简单的代码配置来组装一个符合自己需求的RAG流程。此外，FlashRAG还提供了多种预设的管道，方便用户快速搭建RAG流程。</font>

<font style="color:rgb(88, 90, 115);"></font>

<font style="color:rgb(88, 90, 115);">在管道模块中，我们根据各种RAG任务的操作逻辑，将所有RAG流程分为四种类型：顺序(Sequential)、分支(Branching)、条件(Conditional)和循环(Loop)。目前我们已经实现了8个不同的管道，涵盖了一系列先进的RAG工作。</font>



<font style="color:rgb(88, 90, 115);">FlashRAG的主要目的是帮助研究人员更轻松地进行相关性问答（RAG）研究。传统的RAG研究通常需要手动编写复杂的代码来构建RAG流程，而FlashRAG则提供了丰富的组件和预设的管道，使得用户可以更加高效地搭建RAG流程。此外，FlashRAG还提供了多种数据处理工具，方便用户进行数据过滤和预处理。</font>



## <font style="color:rgb(88, 90, 115);">3. 数据集与语料库</font>
### <font style="color:rgb(88, 90, 115);">3.1 基准数据集</font>
<font style="color:rgb(88, 90, 115);">如表1所示，我们收集并预处理了32个基准数据集，涵盖了RAG工作中使用的大部分数据集。所有数据集都已格式化为统一的JSONL结构。</font>

| <font style="color:rgb(88, 90, 115);">任务</font> | <font style="color:rgb(88, 90, 115);">数据集名称</font> | <font style="color:rgb(88, 90, 115);">知识来源</font> | <font style="color:rgb(88, 90, 115);">训练集大小</font> | <font style="color:rgb(88, 90, 115);">验证集大小</font> | <font style="color:rgb(88, 90, 115);">测试集大小</font> |
| :---: | :---: | :---: | :---: | :---: | :---: |
| <font style="color:rgb(88, 90, 115);">QA</font> | <font style="color:rgb(88, 90, 115);">NQ</font> | <font style="color:rgb(88, 90, 115);">Wiki</font> | <font style="color:rgb(88, 90, 115);">79,168</font> | <font style="color:rgb(88, 90, 115);">8,757</font> | <font style="color:rgb(88, 90, 115);">3,610</font> |
| <font style="color:rgb(88, 90, 115);">QA</font> | <font style="color:rgb(88, 90, 115);">TriviaQA</font> | <font style="color:rgb(88, 90, 115);">Wiki & Web</font> | <font style="color:rgb(88, 90, 115);">78,785</font> | <font style="color:rgb(88, 90, 115);">8,837</font> | <font style="color:rgb(88, 90, 115);">11,313</font> |
| <font style="color:rgb(88, 90, 115);">多跳QA</font> | <font style="color:rgb(88, 90, 115);">HotpotQA</font> | <font style="color:rgb(88, 90, 115);">Wiki</font> | <font style="color:rgb(88, 90, 115);">90,447</font> | <font style="color:rgb(88, 90, 115);">7,405</font> | <font style="color:rgb(88, 90, 115);">/</font> |
| <font style="color:rgb(88, 90, 115);">多跳QA</font> | <font style="color:rgb(88, 90, 115);">2WikiMultiHopQA</font> | <font style="color:rgb(88, 90, 115);">Wiki</font> | <font style="color:rgb(88, 90, 115);">15,000</font> | <font style="color:rgb(88, 90, 115);">12,576</font> | <font style="color:rgb(88, 90, 115);">/</font> |
| <font style="color:rgb(88, 90, 115);">长问题QA</font> | <font style="color:rgb(88, 90, 115);">ELI5</font> | <font style="color:rgb(88, 90, 115);">Reddit</font> | <font style="color:rgb(88, 90, 115);">272,634</font> | <font style="color:rgb(88, 90, 115);">1,507</font> | <font style="color:rgb(88, 90, 115);">/</font> |
| <font style="color:rgb(88, 90, 115);">多选QA</font> | <font style="color:rgb(88, 90, 115);">MMLU</font> | <font style="color:rgb(88, 90, 115);">-</font> | <font style="color:rgb(88, 90, 115);">99,842</font> | <font style="color:rgb(88, 90, 115);">1,531</font> | <font style="color:rgb(88, 90, 115);">14,042</font> |


### <font style="color:rgb(88, 90, 115);">3.2 语料库</font>
<font style="color:rgb(88, 90, 115);">除了数据集，用于检索的语料库（也称为知识库）是实验的另一个重要准备。在FlashRAG中，我们提供了方便的脚本来自动下载和预处理所需的Wikipedia版本。我们还提供了DPR提供的广泛使用的2018年12月20日的Wikipedia转储作为基础资源。</font>

## <font style="color:rgb(88, 90, 115);">4. 实验与结果分析</font>
<font style="color:rgb(88, 90, 115);">本文介绍了作者使用的评估指标和方法来评估基于检索式生成的问答系统（RAG）的质量。作者将这些指标分为两个类别：检索方面和生成方面的指标。</font>

<font style="color:rgb(88, 90, 115);">在检索方面，作者支持四种指标，包括召回率@k、精确度@k、F1@k和平均精度（MAP）。由于在RAG过程中检索到的文档通常缺乏黄金标签（例如相关或不相关的标记），因此作者通过考虑黄金答案是否存在于检索到的文档中作为相关性的指示器来进行评估。其他类型的指标可以通过继承现有的指标并修改计算方法来获得。</font>

<font style="color:rgb(88, 90, 115);">在生成方面，作者支持五种指标，包括词级别F1分数、准确率、BLEU[69]和ROUGE-L[70]。此外，作者还支持评估生成中使用的标记数，以方便分析整个过程的成本。</font>

<font style="color:rgb(88, 90, 115);">为适应自定义评估指标，作者提供了用户可以实现的指标模板。由于该库自动保存执行的中间结果，因此用户可以方便地评估由中间组件产生的结果。例如，用户可能会比较精炼器运行前后的标记数差异，或者多个检索结果之间的精确度差异。</font>

### <font style="color:rgb(88, 90, 115);">4.1 实验设置</font>
<font style="color:rgb(88, 90, 115);">在我们的主要实验中，我们使用最新的LLAMA3-8B-instruct[71]作为生成器，使用E5-base-v2作为检索器，使用2018年12月的Wikipedia数据作为检索语料库。我们在六个常见数据集上进行了实验：Natural Questions(NQ)[38]、TriviaQA[39]、HotpotQA[52]、2WikiMultihopQA[53]、PopQA[40]和WebQuestions[45]。</font>

### <font style="color:rgb(88, 90, 115);">4.2 主要结果</font>
<font style="color:rgb(88, 90, 115);">各种方法的主要结果如表2所示。总的来说，与直接生成的基线相比，RAG方法有显著的改进。使用先进的检索器和生成器的标准RAG是一个强大的基线，在六个数据集上都表现良好。</font>

| <font style="color:rgb(88, 90, 115);">方法</font> | <font style="color:rgb(88, 90, 115);">优化组件</font> | <font style="color:rgb(88, 90, 115);">管道类型</font> | <font style="color:rgb(88, 90, 115);">NQ(EM)</font> | <font style="color:rgb(88, 90, 115);">TriviaQA(EM)</font> | <font style="color:rgb(88, 90, 115);">HotpotQA(F1)</font> | <font style="color:rgb(88, 90, 115);">2Wiki(F1)</font> | <font style="color:rgb(88, 90, 115);">PopQA(F1)</font> | <font style="color:rgb(88, 90, 115);">WebQA(EM)</font> |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| <font style="color:rgb(88, 90, 115);">朴素生成</font> | <font style="color:rgb(88, 90, 115);">-</font> | <font style="color:rgb(88, 90, 115);">顺序</font> | <font style="color:rgb(88, 90, 115);">22.6</font> | <font style="color:rgb(88, 90, 115);">55.7</font> | <font style="color:rgb(88, 90, 115);">28.4</font> | <font style="color:rgb(88, 90, 115);">33.9</font> | <font style="color:rgb(88, 90, 115);">21.7</font> | <font style="color:rgb(88, 90, 115);">18.8</font> |
| <font style="color:rgb(88, 90, 115);">标准RAG</font> | <font style="color:rgb(88, 90, 115);">-</font> | <font style="color:rgb(88, 90, 115);">顺序</font> | <font style="color:rgb(88, 90, 115);">35.1</font> | <font style="color:rgb(88, 90, 115);">58.8</font> | <font style="color:rgb(88, 90, 115);">35.3</font> | <font style="color:rgb(88, 90, 115);">21.0</font> | <font style="color:rgb(88, 90, 115);">36.7</font> | <font style="color:rgb(88, 90, 115);">15.7</font> |
| <font style="color:rgb(88, 90, 115);">AAR[72]</font> | <font style="color:rgb(88, 90, 115);">检索器</font> | <font style="color:rgb(88, 90, 115);">顺序</font> | <font style="color:rgb(88, 90, 115);">30.1</font> | <font style="color:rgb(88, 90, 115);">56.8</font> | <font style="color:rgb(88, 90, 115);">33.4</font> | <font style="color:rgb(88, 90, 115);">19.8</font> | <font style="color:rgb(88, 90, 115);">36.1</font> | <font style="color:rgb(88, 90, 115);">16.1</font> |
| <font style="color:rgb(88, 90, 115);">LongLLMLingua[20]</font> | <font style="color:rgb(88, 90, 115);">精炼器</font> | <font style="color:rgb(88, 90, 115);">顺序</font> | <font style="color:rgb(88, 90, 115);">32.2</font> | <font style="color:rgb(88, 90, 115);">59.2</font> | <font style="color:rgb(88, 90, 115);">37.5</font> | <font style="color:rgb(88, 90, 115);">25.0</font> | <font style="color:rgb(88, 90, 115);">38.7</font> | <font style="color:rgb(88, 90, 115);">17.5</font> |
| <font style="color:rgb(88, 90, 115);">RECOMP[18]</font> | <font style="color:rgb(88, 90, 115);">精炼器</font> | <font style="color:rgb(88, 90, 115);">顺序</font> | <font style="color:rgb(88, 90, 115);">33.1</font> | <font style="color:rgb(88, 90, 115);">56.4</font> | <font style="color:rgb(88, 90, 115);">37.5</font> | <font style="color:rgb(88, 90, 115);">32.4</font> | <font style="color:rgb(88, 90, 115);">39.9</font> | <font style="color:rgb(88, 90, 115);">20.2</font> |
| <font style="color:rgb(88, 90, 115);">SuRe[29]</font> | <font style="color:rgb(88, 90, 115);">管道</font> | <font style="color:rgb(88, 90, 115);">分支</font> | <font style="color:rgb(88, 90, 115);">37.1</font> | <font style="color:rgb(88, 90, 115);">53.2</font> | <font style="color:rgb(88, 90, 115);">33.4</font> | <font style="color:rgb(88, 90, 115);">20.6</font> | <font style="color:rgb(88, 90, 115);">48.1</font> | <font style="color:rgb(88, 90, 115);">24.2</font> |


### <font style="color:rgb(88, 90, 115);">4.3 不同方法的比较分析</font>
<font style="color:rgb(88, 90, 115);">在精炼器优化方法中，所有三种方法都显示出显著的改进。在HotpotQA和2WikiMultihopQA等多跳数据集上，精炼器的表现尤其出色。这可能是因为复杂问题导致文档检索的准确性降低，从而产生更多噪声，需要精炼器优化。</font>



<font style="color:rgb(88, 90, 115);">在生成器优化方法中，Ret-Robust使用带有lora模块的Llama2-13B模型，大大增强了生成器对检索文档的理解，优于其他无需训练的方法。</font>



<font style="color:rgb(88, 90, 115);">优化RAG流程的有效性因数据集而异。在NQ和TriviaQA等较简单的数据集上，FLARE和Iter-RetGen与标准RAG相当或略低于标准RAG。但是，在HotpotQA等需要多步推理的复杂数据集上，它们比基线有显著的改进。这表明适应性检索方法更适合复杂问题，而在简单任务上，它们可能会带来更高的成本，但收益有限。</font>



## <font style="color:rgb(88, 90, 115);">5. FlashRAG的特点与优势</font>


+ <font style="color:rgb(88, 90, 115);">广泛且可定制的模块化RAG框架</font>
+ <font style="color:rgb(88, 90, 115);">预实现的先进RAG算法，目前已实现12个</font>
+ <font style="color:rgb(88, 90, 115);">全面的基准数据集，目前收录32个</font>
+ <font style="color:rgb(88, 90, 115);">高效的RAG辅助脚本，如下载维基百科、构建索引等</font>



## <font style="color:rgb(88, 90, 115);">6. 局限性与未来展望</font>


<font style="color:rgb(88, 90, 115);">我们的工具包目前还存在一些局限性，我们计划在未来逐步改进：</font>



1. <font style="color:rgb(88, 90, 115);">尽管我们努力涵盖许多有代表性的RAG方法，但由于时间和成本的考虑，我们还没有包括所有现有的RAG工作。这可能需要开源社区的贡献。</font>
2. <font style="color:rgb(88, 90, 115);">我们的工具包缺乏对训练RAG相关组件的支持。我们在最初设计时考虑了训练，但考虑到训练方法的多样性以及许多专门用于训练检索器和生成器的存储库的存在，我们没有包括这一部分。未来我们可能会添加一些帮助脚本，为研究人员的训练需求提供一些帮助。</font>



## <font style="color:rgb(88, 90, 115);">7. 总结</font>
<font style="color:rgb(88, 90, 115);">为了应对研究人员在RAG领域复现研究和高开发成本方面面临的挑战，我们引入了一个模块化的RAG工具包FlashRAG。我们的工具包包括全面的RAG基准数据集、先进的RAG方法实现以及用于预处理语料库和多个评估指标的代码。它使研究人员能够轻松地复现现有的RAG方法，开发新的算法，并专注于优化他们的研究。相信FlashRAG能为RAG研究领域提供有力的支持和帮助。</font>



<font style="color:rgb(88, 90, 115);"></font>

<font style="color:rgb(88, 90, 115);"></font>

