另一个 RAG 中的核心组件是生成器，负责将检索到的信息转化为自然流畅的文本。其设计灵感来自传统语言模型，但与常规生成模型相比，RAG 的生成器通过利用检索到的信息来提高准确性和相关性。

在 RAG 中，生成器的输入不仅包括传统的上下文信息，还包括通过检索器获得的相关文本片段。这使得生成器能更好地理解问题背后的上下文，并生成更丰富信息的回应。

此外，生成器由检索到的文本引导，以确保生成内容与检索信息之间的一致性。正是输入数据的多样性促使了一系列针对生成阶段的目标努力，所有这些都旨在更好地适应来自查询和文档的输入数据的大型模型。我们将通过后检索处理和微调的方面介绍生成器。

## 如何通过后检索处理提升检索结果？
在未经调整的大型语言模型方面，大多数研究依赖于像 GPT-4这样的公认的大型语言模型，利用它们强大的内部知识进行全面的文档知识检索。

然而，这些大型模型的固有问题，如上下文长度限制和对冗余信息的易感性，仍然存在。为了缓解这些问题，一些研究在后检索处理方面做出了努力。

后检索处理指的是对检索器从大型文档数据库中检索到的相关信息进行进一步处理、过滤或优化的过程。其主要目的是提高检索结果的质量，以更好地满足用户需求或用于后续任务。可以理解为对检索阶段获得的文档进行再处理的过程。后检索处理的操作通常涉及信息压缩和结果重排。

### 信息压缩
尽管检索器可以从庞大的知识库中获取相关信息，我们仍然面临着处理检索文档中大量信息的挑战。一些现有研究尝试通过增加大型语言模型的上下文长度来解决这个问题，但当前的大型模型仍然面临上下文限制。

因此，在某些情况下，信息压缩是必要的。简而言之，信息压缩的重要性主要体现在以下几个方面：降低噪音、应对上下文长度限制和增强生成效果。

另一项研究选择进一步精简文档数量，目的是通过减少检索到的文档数量来提高模型的答案准确性。例如“过滤-排序”范式，结合了大型语言模型（LLMs）和小型语言模型（SLMs）的优势。

在这一范式中，SLMs 充当过滤器，而 LLMs 作为重新排序代理。通过提示 LLMs 重排 SLMs 识别的困难样本部分，研究结果表明在各种信息提取（IE）任务中取得了显著的改进。

### 重排
排序模型的关键作用在于优化检索器检索到的文档集。当添加额外上下文时，LLMs 的性能会出现退化，而重排提供了解决这一问题的有效解决方案。

核心思想涉及重新排列文档记录，将最相关的项目置于顶部，从而将文档总数减少到固定数量。这不仅解决了检索过程中可能遇到的上下文窗口扩展问题，还有助于提高检索效率和响应速度作为排序的一部分引入上下文压缩旨在仅基于给定查询上下文返回相关信息。

这种方法的双重意义在于通过减少单个文档的内容和过滤整个文档，集中显示检索结果中最相关的信息。

因此，排序模型在整个信息检索过程中发挥优化和精炼作用，为后续的 LLM 处理提供更有效、更准确的输入。

## 如何优化生成器以适应输入数据？
在 RAG 模型中，生成器的优化是架构的一个关键组成部分。生成器的任务是利用检索到的信息生成相关文本，从而提供模型的最终输出。优化生成器的目标是确保生成的文本既自然又有效地利用检索到的文档，以更好地满足用户的查询需求。

在典型的大型语言模型（LLM）生成任务中，输入通常是一个查询。在 RAG 中，主要区别在于输入不仅包括查询，还包括检索器检索到的各种文档（结构化/非结构化）。

引入额外信息可能对模型的理解产生重大影响，特别是对于较小的模型。在这种情况下，微调模型以适应查询+检索文档的输入变得尤为重要。

具体来说，在将输入提供给微调后的模型之前，通常会对检索器检索到的文档进行后检索处理。需要注意的是，RAG 中微调生成器的方法本质上类似于 LLMs 的一般微调方法。

