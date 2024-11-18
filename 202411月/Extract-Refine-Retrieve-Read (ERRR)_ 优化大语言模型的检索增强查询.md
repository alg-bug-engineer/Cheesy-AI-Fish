> 原文：<font style="color:#000000;">Query Optimization for Parametric Knowledge Refinement in Retrieval-Augmented Large Language Models</font>
>

## 一、研究背景与挑战
在自然语言处理领域,大语言模型(LLMs)取得了显著进展。这些模型通过在海量语料上进行训练,展现出理解人类文本和生成高质量响应的卓越能力。然而,大语言模型存在一个重要的局限性 - 它们难以准确捕捉信息的动态更新。因为这些模型是在静态语料库上预训练的,当面对最新或不常见的信息时,它们往往会生成过时或不准确的内容,这种现象通常被称为"幻觉"(hallucination)。

为了解决这个问题,检索增强生成(Retrieval-Augmented Generation, RAG)技术应运而生。RAG通过信息检索系统来整合外部知识,并利用模型的上下文学习能力来生成更准确的响应。但是,RAG系统本身也带来了新的挑战,其中一个关键问题是原始查询与生成最佳响应所需信息之间存在差距。

例如,当外部文档集包含三个独立的段落(Passage A、B、C)时,每个段落包含独特的知识组件(分别是知识x、y、z)。用户想要获取知识z,但如果用户输入的查询表述不够准确,系统可能会检索到Passage A或B,而不是理想的Passage C,这就限制了模型生成最佳响应的能力。这种差距被称为RAG系统中的预检索差距(pre-retrieval gap)。

## 二、ERRR框架设计
![](https://cdn.nlark.com/yuque/0/2024/png/406504/1731890501789-9e345c20-21b8-42a1-9738-bef7945869f8.png)

Extract-Refine-Retrieve-Read (ERRR)框架的形式化定义可以从检索增强生成任务开始。给定一个输入查询q，一组包含准确信息以回答查询q的理论黄金文档集D，以及一个真实答案a，可以将基本任务表示为：

$ LLM(D, q | θ) = a $

其中θ表示LLM的参数知识。在实际应用中，通常使用检索函数R从外部知识库中获取文档R(q)，因此检索增强系统的输出可以表示为：

$ LLM(R(q), q | θ) $

这两个表达式之间的差异揭示了检索增强系统中的预检索差距问题：

$ LLM(R(q), q | θ) ≠ LLM(D, q | θ) $

ERRR框架通过引入参数知识提取函数E和专门的查询优化函数f'来解决这个问题：

$ LLM(R(f'(C, q)), q | θ) $

其中：

$ C = E(q | θ) $

在参数知识提取阶段，ERRR采用直接提示策略。通过向模型提供特定的提示模板，例如：

```plain
Template: "Generate a background document from web to answer the given question. {query}"
```

这个阶段的输出是一个包含模型当前参数知识的伪上下文文档。这个文档虽然可能包含一些不准确信息，但为后续的查询优化提供了重要的上下文信息基础。

![](https://cdn.nlark.com/yuque/0/2024/png/406504/1731890519633-8b03aad4-511c-4488-9469-ec7d73cd8b4d.png)

查询优化器可以使用如下模板来生成优化查询：

```plain
Template: "Address the following questions based on the contexts provided. 
          Identify any missing information or areas requiring validation, 
          especially if time-sensitive data is involved. Then, formulate 
          several specific search engine queries to acquire or validate 
          the necessary knowledge. 
          Context: {Parametric_Knowledge} 
          Question: {query} 
          Queries:"
```

在检索阶段，系统支持两种检索方式，可以形式化表示为：

$ R_{web}(q) = WebSearchEngine(q) $

$ R_{dense}(q) = TopK(sim(Encode(q), D_{embedded})) $

其中$ R_{web} $表示网络搜索检索器，$ R_{dense} $表示密集检索器，$ sim $是相似度计算函数，通常使用L2距离。

对于每个具体的任务，如AmbigQA数据集，系统使用特定的提示模板来确保输出格式的一致性：

```plain
Template: "Answer the question in the following format, end the answer with '**'. 
          {demonstration} Question: {doc} {query} 
          Answer:"
```

整个框架的优化目标可以表示为最小化预检索差距：

$ min_{\theta'} \mathbb{E}_{q,a}[L(LLM(R(f'(E(q|θ), q)), q | θ), a)] $

其中L表示损失函数，用于衡量生成答案与真实答案之间的差距。

在实际应用中，当处理类似"Stories USA starred which actor and comedian from The Office?"这样的查询时，框架会首先通过参数知识提取生成一个包含Steve Carell相关信息的文档，然后优化器会生成多个验证查询：

1. "actor and comedian from The Office in Stories USA"
2. "Steve Carell role in Stories USA"

这些优化查询不仅验证了演员的基本信息，还特别关注了在具体作品中的角色信息，从而能够获取更准确的答案。

通过这种精心设计的数学框架和实现细节，ERRR能够有效地解决预检索差距问题，提供更准确的答案。每个组件都经过严格的形式化定义，确保了系统的可靠性和可复现性。

Extract-Refine-Retrieve-Read (ERRR)框架是一种旨在提升检索增强大语言模型性能的系统架构。该框架通过融合参数知识提取、查询优化、信息检索和答案生成四个关键环节，形成了一个完整的技术解决方案。该框架包含四个核心步骤:

### 2.1 参数知识提取(Parametric Knowledge Extraction)
在参数知识提取阶段，ERRR框架采用了一种直接的策略。它会提示大语言模型生成一个包含所有背景信息的伪上下文文档。这个做法的灵感来源于GenRead研究，其核心思想是利用模型已有的参数知识。值得注意的是，这些生成的伪文档虽然可能包含一些不准确信息，但它们为后续的查询优化提供了重要的上下文信息基础。例如，当询问某个历史事件时，模型可能会生成一个包含相关背景、人物和时间线的概述文档。

### 2.2 查询优化(Query Optimization)
在查询优化环节，ERRR使用专门的语言模型作为查询优化器。这个优化器的主要任务是生成一个或多个经过优化的查询语句。这些查询语句主要服务于两个目标：一是验证从参数知识中提取的信息，二是补充可能缺失的信息。特别需要强调的是，优化器会特别关注时效性信息的验证。例如，如果原始查询涉及某位政治人物的现任职位，优化器会生成专门用于验证该职位最新状态的查询。

### 2.3 检索(Retrieval)
检索阶段展现了ERRR框架的灵活性和适应性。该框架支持两种不同类型的检索系统：一种是黑盒式的网络搜索工具，如Brave搜索引擎；另一种是基于本地密集检索的系统，如Dense Passage Retrieval (DPR)。在实际应用中，网络搜索工具能够提供最新的信息，而本地密集检索系统则可以提供更稳定和可控的搜索结果。这两种检索方式分别适用于不同的应用场景。

### 2.4 生成(Generation)
最后的生成阶段采用了一种直接但有效的方法。系统使用大语言模型作为阅读器，将检索到的文档与原始查询结合起来生成最终答案。为了确保输出格式的一致性，每个数据集都配备了特定的少量示例（通常是1-3个）作为提示。这些示例帮助模型理解任务的具体要求。比如在某些问答任务中，答案需要保持简洁，通常只包含一个或几个词；而在其他任务中，可能需要更详细的解释。



ERRR框架的核心创新在于它建立了一个完整的知识验证和补充机制。即使在参数知识提取阶段生成的伪文档中包含不准确信息，后续的查询优化和检索步骤也能够通过外部知识的获取来纠正这些错误。这种设计不仅提高了答案的准确性，还增强了系统的可靠性。

框架的实现过程中特别注意了时效性信息的处理。例如，当遇到与时事相关的查询时，系统会生成专门用于验证最新信息的查询语句。这种设计帮助系统在处理动态变化的信息时保持高准确度。同时，通过整合本地密集检索和网络搜索，系统能够在不同类型的知识获取任务中保持良好的表现。

通过这种精心设计的流程，ERRR框架能够有效地弥合预检索差距，提供更准确的答案。框架的每个组件都经过深思熟虑的设计，既保证了系统的整体性能，又维持了较高的灵活性和可扩展性。这种设计理念使ERRR能够在各种复杂的问答场景中展现出优秀的性能。

## 四、实验验证
ERRR框架的实验评估采用了系统化的方法，通过多维度的测试来验证其有效性。实验设计主要围绕三个核心问题展开：框架的整体性能、适应性以及计算效率。

### 数据集选择与评估指标
实验选择了三个具有代表性的开放域问答数据集：

1. AmbigQA数据集：这是Natural Questions数据集的消歧变体。研究者选择了测试集的前1000个样本进行评估。这个数据集的特殊之处在于其问题具有多重解释的可能性，例如"谁是奥斯卡最佳导演"这样的问题可能需要指定具体年份。这种特性使得它特别适合测试模型处理模糊查询的能力。
2. PopQA数据集：该数据集包含了997个测试样本，其特点是问题聚焦于较少见的知识主题。例如，它可能会询问一些非主流的历史事件或者相对冷门的科技发展。这类问题对模型的知识覆盖范围提出了更高的要求。
3. HotpotQA数据集：这个数据集的特点是包含需要多步推理的复杂问题。例如，一个问题可能需要先找到某个人物的出生地，然后再查询这个地方的特定历史事件。这种多跳推理的特性使其成为测试模型推理能力的理想选择。

所有实验采用两个关键指标：精确匹配分数(EM)和F1分数。对于复杂问题，这两个指标能够分别反映完全正确的答案比例和部分正确的情况。

### 基线方法对比
![](https://cdn.nlark.com/yuque/0/2024/png/406504/1731890543520-f03f0a1f-c642-4f0d-b574-1e323d08e413.png)

研究者设计了七种不同的对比方法：

1. Direct方法：直接使用GPT-3.5-Turbo回答问题，这代表了纯参数知识的基准线。
2. RAG：经典的检索增强生成框架，使用原始查询进行检索。
3. ReAct：这是一个修改版的RAG框架，它通过交替使用推理和行动来创建更连贯的方法。
4. Frozen RRR：使用固定配置的Rewrite-Retrieve-Read框架。
5. Trainable RRR：可训练的RRR框架，使用T5-large模型进行监督微调。
6. Frozen ERRR：作者提出的框架的固定配置版本。
7. Trainable ERRR：框架的可训练版本。

实验结果显示出一些深入的见解。以AmbigQA数据集为例：

```plain
数据分析示例：
Direct方法: EM=0.391, F1=0.4996
Frozen ERRR: EM=0.4815, F1=0.5823
Trainable ERRR: EM=0.4975, F1=0.5988
```

这组数据揭示了几个关键发现：

1. 参数知识的局限性：Direct方法的较低表现（EM=0.391）说明纯粹依赖模型的参数知识是不足的。
2. 检索增强的效果：ERRR方法在EM和F1上都显著优于Direct方法，证实了检索增强的必要性。
3. 训练的重要性：Trainable ERRR相比Frozen ERRR有更好的表现，表明模型通过训练可以更好地适应特定任务。

### 深入性能分析
研究者还进行了一系列深入分析。在检索系统选择方面，实验对比了两种不同的检索器：

1. Brave搜索API：作为网络搜索工具，其优势在于能够获取最新信息，但API调用成本较高。
2. WikiDPR：这是一个本地密集检索系统，基于2018年12月20日的维基百科数据，包含2100万个段落。

性能对比显示了一个有趣的现象：

```plain
使用Brave搜索API时：
Frozen ERRR: EM=0.4815, F1=0.5823

使用WikiDPR时：
Frozen ERRR: EM=0.448, F1=0.5473
```

这种差异揭示了实时网络搜索相比静态知识库的优势，特别是在处理时效性信息时。

### 计算效率分析
研究者还特别关注了计算效率问题。在HotpotQA数据集的200个随机问题上进行的效率测试显示：

```plain
方法比较：
Frozen ERRR: 成本=$0.62, 延迟=148s
Trainable ERRR: 成本=$0.53, 延迟=140s
ReAct: 成本=$1.05, 延迟=202s
Self-RAG: 成本=$1.65, 延迟=270s
```

这组数据揭示了ERRR框架在保持高性能的同时，实现了更好的计算效率。特别是Trainable ERRR通过使用预先微调的查询优化器，避免了额外的GPT-3.5-Turbo调用，从而进一步降低了成本。

### 案例分析
研究者通过具体案例深入分析了框架的工作机制。以"Stories USA starred which actor and comedian from The Office?"这个问题为例：

1. RRR方法生成的查询："actor comedian The Office Stories USA cast"
+ 结果：错误回答"Ricky Gervais"
2. ERRR方法的处理过程：
+ 首先提取参数知识，识别出Steve Carell的相关信息
+ 生成优化查询：
    - "actor and comedian from The Office in Stories USA"
    - "Steve Carell role in Stories USA"
+ 结果：正确回答"Steven John Carell"

这个案例展示了ERRR框架如何通过结合参数知识和优化查询来提升回答的准确性。即使在参数知识可能不完全准确的情况下，优化的查询策略也能帮助系统获取到正确的信息。

这些详细的实验分析不仅验证了ERRR框架的有效性，还揭示了框架在不同场景下的适应性和效率优势。通过多维度的评估和深入的案例分析，研究者展示了ERRR框架在实际应用中的潜力和价值。

## 五、应用案例
以下是一个具体的案例来说明ERRR的工作原理:

原始问题:"Stories USA starred which actor and comedian from The Office?"

RRR的重写查询仅是:"actor comedian The Office Stories USA cast"

而ERRR的处理流程更加深入:

1. 首先提取参数知识,得知Steve Carell可能出演该片
2. 生成优化查询:"actor and comedian from The Office in Stories USA"和"Steve Carell role in Stories USA"
3. 最终成功得到正确答案:"Steven John Carell"

这个案例展示了ERRR如何通过结合参数知识和优化查询来提升答案的准确性。

