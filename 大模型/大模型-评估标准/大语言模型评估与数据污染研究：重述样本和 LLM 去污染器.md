# 大语言模型评估与数据污染研究：重述样本和 LLM 去污染器
> 原文：《Rethinking Benchmark and Contamination for Language Models with Rephrased Samples》
## 1. 研究背景与动机
### 1.1 大语言模型评估面临的挑战
随着大语言模型（Large Language Models, LLMs）的快速发展，评估这些模型的能力变得越来越具有挑战性。传统的评估方法可能无法准确反映模型在真实世界任务中的表现，特别是当模型性能接近或超过人类水平时。

### 1.2 数据污染问题的严重性
研究发现，许多流行的基准测试数据集可能已经被包含在模型的预训练或微调数据集中，这种现象被称为"数据污染"。例如：

+ Llama-2 的分析显示，超过 10% 的 MMLU（大规模多任务语言理解）测试样本被高度污染。
+ GPT-4 的技术报告指出，25% 的 HumanEval（编程能力评估数据集）已经在其训练数据中被污染。

这种污染会导致模型在测试中表现异常优秀，但这并不能真实反映模型的实际能力。

### 1.3 现有去污染方法的局限性
目前最常用的去污染方法包括 n-gram 重叠检测和嵌入相似度搜索。然而，这些方法存在明显的局限性：

1. **n-gram 重叠检测**：
    - 原理：检查测试样本中的 n 个连续词（或字符）是否出现在训练数据中。
    - 优点：简单快速，易于实现。
    - 缺点：容易出现高假阴性率，即无法检测到稍有变化的污染样本。
    - 示例：如果训练集中有 "The cat sat on the mat"，测试集中有 "The dog sat on the mat"，10-gram 检测可能无法发现这种相似性。
2. **嵌入相似度搜索**：
    - 原理：使用预训练模型（如 BERT）将文本转换为向量，然后计算向量间的相似度。
    - 优点：能捕捉更多语义信息，不仅限于字面匹配。
    - 缺点：难以设定合适的相似度阈值，容易导致高假阳性或假阴性率。
    - 示例：如果设置阈值为 0.8，"I love apples" 和 "I adore apples" 可能被判定为不同，而 "The sky is blue" 和 "The ocean is vast" 可能被误判为相似。

这些挑战促使研究者们探索更有效的方法来评估大语言模型和解决数据污染问题。

## 2. 研究方法
### 2.1 提出"重述样本"(rephrased samples)概念
重述样本是指与原始样本具有相同语义但难以被现有污染检测方法发现的变体。生成方法包括：

+ 对于文本类基准：改变词序或使用同义词，保持语义不变。
+ 对于代码类基准：改变编码风格、命名约定和实现方式，但保持功能不变。
+ 使用高质量语言模型（如 GPT-4）来生成重述版本。

示例（GSM-8k 数据集）：

原始问题：

`Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?`

重述后：

`Janet's ducks produce 16 eggs each day. She consumes three of them for her morning meal and uses four to bake muffins for her friends daily. The remaining eggs are sold at the daily farmers' market for $2 per egg. What is the daily amount in dollars that she earns at the farmers' market?`

### 2.2 设计新的 LLM 去污染器(LLM decontaminator)
研究者提出了一种两步法的新型去污染方法：

1. 使用嵌入相似度搜索找出与测试样本最相似的 top-k 个训练样本。
2. 使用高级语言模型（如 GPT-4）评估这些样本对是否实质上相同。

具体算法如下：

```plain
def Decontaminate(TrainSet, TestSet, k, Template):
    Contamination = ∅
    for t in TestSet:
        for c in TopKSimilarity(TrainSet, t, k):
            s = LLMDetector(Template, t, c)
            if s == True:
                Contamination = Contamination ∪ {(t, c)}
    return Contamination
```

这种方法结合了嵌入相似度搜索的广泛检测能力和大语言模型的精确判断能力，能够更准确地识别重述样本。

### 2.3 在多个基准测试集上进行实验验证
研究者在多个广泛使用的基准测试集上进行了实验，包括：

+ MMLU（多任务语言理解）
+ HumanEval（编程能力评估）
+ GSM-8k（数学问题求解）

他们使用重述样本训练模型，然后评估模型在原始测试集上的表现，以验证重述样本的影响。

## 3. 主要发现
### 3.1 重述样本可以轻易绕过现有去污染方法
实验表明，重述样本能够成功逃过 n-gram 重叠检测和标准的嵌入相似度搜索。下表展示了不同检测方法在 MMLU 数据集上的 F1 分数：

![](https://cdn.nlark.com/yuque/0/2024/png/406504/1728975350089-126de060-c612-4ad5-9e81-8124c9e7b47e.png)

### 3.2 包含重述样本的训练会导致模型在测试集上表现异常出色
研究者发现，使用重述样本训练的模型在原始测试集上能够达到惊人的性能：

![](https://cdn.nlark.com/yuque/0/2024/png/406504/1728975350165-fa5ed7e4-6956-46ae-9f44-2a41ba314fe7.png)

这些性能提升远远超过了模型的真实能力，甚至达到或超过了 GPT-4 的水平。

### 3.3 LLM 去污染器能有效检测重述样本
与其他方法相比，新提出的 LLM 去污染器在检测重述样本时表现出色：

+ 在 MMLU 数据集上，LLM 去污染器的 F1 分数始终保持在 0.94 以上。
+ 在 HumanEval 数据集上，LLM 去污染器的 F1 分数在 0.974-0.995 之间，而 10-gram 重叠检测的 F1 分数为 0。

## 4. 真实数据集分析
### 4.1 在多个流行数据集中发现未知的测试集重叠
研究者将 LLM 去污染器应用于多个广泛使用的真实世界数据集，发现了许多先前未知的测试集重叠：

![](https://cdn.nlark.com/yuque/0/2024/png/406504/1728975350201-5394e5a7-f749-41c3-871c-f7cc5f8c319a.png)

这些发现表明，即使是被广泛使用的开源数据集也存在严重的污染问题。

### 4.2 在 LLM 生成的合成数据中也存在污染风险
研究者还分析了由 GPT-3.5 生成的 CodeAlpaca 数据集，发现其中包含 21 个 HumanEval 重述样本，占比 12.8%。这表明使用 LLM 生成的合成数据进行训练也可能带来潜在的污染风险。

示例（CodeAlpaca 中的重述样本）：

原始 HumanEval 测试：

```plain
def sum_to_n(n: int):
    """sum_to_n is a function that sums numbers from 1 to n.
    >>> sum_to_n(30)
    465
    >>> sum_to_n(100)
    5050
    >>> sum_to_n(5)
    15
    >>> sum_to_n(10)
    55
    >>> sum_to_n(1)
    1
    """
    return sum(range(n + 1))
```

CodeAlpaca 中的重述版本：

```plain
"""
Create a code that summation of all numbers between 1 to n.
"""
def sum_all_nums(n):
    res = 0
    for i in range(1, n+1):
        res += i
    return res

print(sum_all_nums(n)) # 15
```

## 5. 讨论与建议
### 5.1 呼吁采用更强大的去污染方法
鉴于重述样本带来的挑战，研究者呼吁社区在使用公共基准评估 LLM 时采用更强大的去污染方法，如本文提出的 LLM 去污染器。

### 5.2 提议开发"一次性考试"式的新鲜基准测试
为了从根本上解决污染问题，研究者建议开发类似 Codeforces 和 Kaggle 比赛的新鲜、一次性的评估任务。这样可以确保测试数据不会被提前泄露或包含在训练集中。

### 5.3 探讨污染定义的边界情况
研究者指出，精确定义何为污染仍然具有挑战性。例如，在 GSM-8k 数据集中，有些训练样本和测试样本仅在数字上有差异：

训练集样本：

`When Diane turns 30, she will be half the age of Alex and twice as old as Allison. Diane is 16 years old now. What is the sum of the ages of Alex and Allison now?`

测试集样本：

`Emil is 19 years old now. When he turns 24, he will be half the age of his dad but twice as old as his brother. What is the sum of the ages of his dad and his brother now?`

这种情况是否应该被视为污染，仍需进一步讨论。

## 6. 结论与未来工作
本研究揭示了大语言模型评估中的一个重要问题：即使是微小的数据变化也可能导致严重的基准污染。研究者提出的 LLM 去污染器为解决这一问题提供了有效工具。

未来的工作方向可能包括：

1. 进一步完善 LLM 去污染器，提高其效率和准确性。
2. 探索如何在没有训练数据访问权限的情况下检测污染。
3. 开发更多动态、一次性的评估方法，以避免静态基准测试的局限性。
4. 深入研究合成数据在训练中的应用，以及如何降低其带来的污染风险。

这项研究为大语言模型的评估和数据清洁提供了新的视角和方法，对推动该领域的发展具有重要意义。

