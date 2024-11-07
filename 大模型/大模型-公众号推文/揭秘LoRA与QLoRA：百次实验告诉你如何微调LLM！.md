LoRA（低秩适应）是目前应用最广泛、参数效率最高的自定义大型语言模型（LLM）微调技术之一。本文不仅介绍了使用QLoRA节省内存的方法，还讨论了选择最佳LoRA设置的实用技巧，为有兴趣应用此技术的读者提供了实践洞见。

## 如何充分利用LoRA
过去几个月里，我进行了数百次甚至上千次涉及LoRA的实验。几周前，我花时间深入研究了一些超参数的选择。

这篇文章更像是按时间顺序呈现的实验日记。我希望它对一些人有用。具体来说，我想回答关于QLoRA价值的问题，是否应该用SGD替换AdamW，潜在的使用调度器，以及如何调整LoRA超参数。

关于实验方面有很多内容需要讨论，因此我会简短介绍LoRA。

简而言之，LoRA（Hu等人，2021年提出的低秩适应）通过向模型添加少量可训练参数，同时保持原始模型参数不变，实现了功能。

LoRA通过将一个大的权重矩阵分解为两个较小的权重矩阵，如下图所示，以更高的参数效率近似实现完全的有监督微调。

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1703819031604-56b9e451-edd9-4e9b-9181-019891926c02.png)

### 评估任务和数据集
本文的重点是选择最优设置。为了保持合理的范围，我固定了数据集，仅专注于大型语言模型（LLM）的有监督指令微调。

对于模型评估，我从Eleuther AI的评估工具包中选取了一小部分任务，包括TruthfulQA、BLiMP因果关系、MMLU全球事实，以及两位数（算术2ds）和四位数（算术4ds）的简单算术任务。

在每个基准测试中，模型性能得分在0到1之间标准化，其中1为满分。TruthfulQA报告两个得分，定义如下：

+ MC1（单一真实）：给定一个问题和4-5个答案选项，选择唯一正确的答案。模型的选择是它分配给紧随问题之后最高对数概率完成的答案选项，独立于其他答案选项。分数是所有问题的简单准确率。
+ MC2（多重真实）：给定一个问题和多个真/假参考答案，得分是分配给一组真实答案的标准化总概率。

> 作为参考，175B GPT-3模型的TruthfulQA MC1和MC2值分别为0.21和0.33。
>

下面是两个例子，用以说明算术2ds和算术4ds之间的区别：

+ 算术2ds：“59减38是多少？”答案：“21”。
+ 算术4ds：“2762加2751是多少？”答案：“5513”。

如上所述，固定了数据集，使用了广为研究或常用的Alpaca数据集进行有监督指令微调。当然，还有许多其他用于指令微调的数据集，包括LIMA、Dolly、LongForm、FLAN等。然而，未来的研究中，探索在多个数据集和数据集组合上的训练将是一个有趣的话题。

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1703819034201-dbe805fd-47fc-4e85-81d4-3fb5998cd0fe.png)

数据集样例数据如下图所示：

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1703819034524-417db6a3-578a-4d6b-9c50-9a7c22d4fee4.png)

### 代码框架
> Lit-GPT:[https://github.com/Lightning-AI/lit-gpt](https://github.com/Lightning-AI/lit-gpt)
>

我在这篇文章中使用的自定义大型语言模型（LLM）微调代码基于开源的Lit-GPT仓库。为了使文章的前言简洁，我不会深入讨论使用细节，但你可以在Lit-GPT教程部分找到更详细的指南。

简要来说，使用方法如下：

1. 克隆相关仓库和安装相关依赖

```python
git clone https://github.com/Lightning-AI/lit-gpt

cd lit-gpt

pip install -r requirements.txt
```

2. 下载模型ckpt文件

```python
python scripts/download.py \
--repo_id mistralai/Mistral-7B-Instruct-v0.1
# there are many other supported models
```

```python
python scripts/convert_hf_checkpoint.py \
--checkpoint_dir checkpoints/mistralai/Mistral-7B-Instruct-v0.1
```

3. 数据准备

```python
python scripts/prepare_alpaca.py \
--checkpoint_dir checkpoints/mistralai/Mistral-7B-Instruct-v0.1
```

```python
# or from a custom CSV file
python scripts/prepare_csv.py \
--csv_dir MyDataset.csv \
--checkpoint_dir checkpoints/mistralai/Mistral-7B-Instruct-v0.1
```

4. 进行监督微调

```python
python finetune/lora.py \
--checkpoint_dir checkpoints/mistralai/Mistral-7B-Instruct-v0.1/ \
--precision bf16-true
```

5. 将Lora权重合并到原始模型上

```python
python scripts/merge_lora.py \
--checkpoint_dir "checkpoints/mistralai/Mistral-7B-Instruct-v0.1" \
--lora_path "out/lora/alpaca/Mistral-7B-Instruct-v0.1/lit_model_lora_finetuned.pth" \
--out_dir "out/lora_merged/Mistral-7B-Instruct-v0.1/"

cp checkpoints/mistralai/Mistral-7B-Instruct-v0.1/*.json \
out/lora_merged/Mistral-7B-Instruct-v0.1/
```

6. 效果评估

```python
python eval/lm_eval_harness.py \
--checkpoint_dir "out/lora_merged/Mistral-7B-Instruct-v0.1/" \
--eval_tasks "[arithmetic_2ds, ..., truthfulqa_mc]" \
--precision "bf16-true" \
--batch_size 4 \
--num_fewshot 0 \
--save_filepath "results.json"
```

7. 模型使用

```python
python chat/base.py \ 
--checkpoint_dir "out/lora_merged/Mistral-7B-Instruct-v0.1/"
```

## 选择一个好的基础模型 
首先，我需要为LoRA实验选择一个合适的基础模型。在此，我关注的是那些尚未经过指令微调的模型：phi-1.5 1.3B、Mistral 7B、Llama 2 7B、Llama 2 13B和Falcon 40B。值得注意的是，所有实验都是在单个A100 GPU上运行的。

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1703819031816-e25c2a17-f031-403f-a01f-5b0d8761341a.png)

从上表我们可以看出，Mistral 7B模型在数学基准测试上表现非常出色。与此同时，考虑到其相对较小的规模，phi-1.5 1.3B模型在TruthfulQA MC2上展现了令人印象深刻的性能。出于某种原因，Llama 2 13B在算术基准测试中表现欠佳，而较小的Llama 2 7B在这方面的表现显著优于它。

由于研究人员和从业者目前推测phi-1.5 1.3B和Mistral 7B可能已在基准测试数据上进行了训练，所以我选择不在我的实验中使用它们。此外，我认为选择剩余模型中最小的一个将在保持较低硬件要求的同时提供最大的改进空间。因此，本文的剩余部分将聚焦于Llama 2 7B。

### 评估LoRA的默认设置
首先，我使用以下默认设置评估了LoRA的微调（这些设置可以在finetune/lora.py脚本中更改）：

> Lit-GPT: [https://github.com/Lightning-AI/lit-gpt](https://github.com/Lightning-AI/lit-gpt)
>

```python
# Hyperparameters
learning_rate = 3e-4
batch_size = 128
micro_batch_size = 1
max_iters = 50000  # train dataset size
weight_decay = 0.01
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_query = True
lora_key = False
lora_value = True
lora_projection = False
lora_mlp = False
lora_head = False
warmup_steps = 100
```

（请注意，批处理大小为128，但我们使用带有1个微批处理的梯度累积来节省内存；这导致了与常规使用128批处理大小的训练相同的训练轨迹。）

这个配置训练了4,194,304个LoRA参数，总共有6,738,415,616个可训练参数，并且在我使用单个A100的机器上大约花费了1.8小时。最大内存使用量为21.33 GB。

为了衡量差异，我重复进行了三次实验，观察了不同运行之间性能的波动。

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1703819034545-b82f2a24-5723-4162-91dd-539df395d1a3.png)

正如我们在上表中看到的，不同运行之间的性能非常一致和稳定。同样值得注意的是，LoRA默认模型在算术任务上表现非常差，但这可能是因为据我所知，Alpaca数据集并没有（或很少有）算术任务。

此外，我还研究了Meta使用RLHF对7B Llama 2版本进行指令微调后的模型。根据下表，Meta的Llama 2 Chat模型在算术性能上也更差。然而，Chat模型在其他基准测试（除BLiMP外）上有了显著改进，我们可以将其作为我们想要通过LoRA微调接近的参考。

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1703819032669-84b85f4f-94c8-4f1c-9ff8-e1221d546245.png)

### 使用QLoRA节省内存
在我们开始调整LoRA超参数之前，我想探索QLoRA（Dettmers等人提出的流行的量化LoRA技术）在模型性能和内存节省之间的权衡。

我们可以通过在Lit-GPT中使用–quantize标志（这里使用4位正常浮点类型）来启用QLoRA，如下所示：

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1703819034063-bc1ec04e-912c-4d5e-967c-99a89ee6642f.png)

此外，我还尝试了4位浮点精度作为对照。以下是对训练时间和最大内存使用量的影响：

默认LoRA（使用bfloat-16）：

+ 训练时间：6685.75秒
+ 内存使用：21.33 GB

通过–-quantize “bnb.nf4”启用的QLoRA：

+ 训练时间：10059.53秒
+ 内存使用：14.18 GB

通过–quantize “bnb.fp4”启用的QLoRA：

+ 训练时间：9334.45秒
+ 内存使用：14.19 GB

我们可以看到，QLoRA将内存需求减少了近6 GB。然而，代价是训练时间延长了30%，这是由于额外的量化和反量化步骤所致。

接下来，让我们看看QLoRA训练如何影响模型性能：

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1703819034312-db77b99b-50e5-4179-8117-5a3539084683.png)

从上表中可以看出，与常规QLoRA相比，QLoRA对模型性能确实有一些影响。模型在算术基准测试中有所改进，但在MMLU全球事实基准测试中有所下降。

由于内存节省相当可观（这通常会超过较长的训练时间，因为它允许用户在较小的GPU上运行模型），我将在本文的其余部分使用QLoRA。

### 学习率调度器和SGD
我在之前的所有实验中都使用了AdamW优化器，因为它是LLM训练的常见选择。然而，众所周知，Adam优化器可能非常占用内存。这是因为它为每个模型参数引入并跟踪两个额外的参数（动量m和v）。大型语言模型（LLM）有许多模型参数；例如，我们的Llama 2模型有70亿个模型参数。

本节将探讨用SGD优化器替换AdamW是否值得。然而，对于SGD优化器，引入学习率调度器尤为重要。我选择了一个余弦退火调度，它在每次批量更新后降低学习率。

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1703819035607-ca7d1deb-2e19-4e43-a7a4-978603579564.png)

不幸的是，将AdamW替换为SGD只节省了少量内存。

+ AdamW：14.18 GB
+ SGD：14.15 GB

这可能是因为大部分内存被用于大型矩阵乘法，而不是存储额外的参数。

但这种小差异或许是意料之中的。在当前选择的LoRA配置（r=8）下，我们有4,194,304个可训练参数。如果Adam为每个模型参数添加2个额外值，并且以16位浮点数存储，那么我们有4,194,304 * 2 * 16位 = 134.22兆比特 = 16.78兆字节。

当我们将LoRA的r增加到256时，我们可以观察到更大的差异，这一点我们稍后会做。在r=256的情况下，我们有648,871,936个可训练参数，使用上述同样的计算方法，相当于2.6 GB。实际测量结果显示有3.4 GB的差异，可能是由于存储和复制优化器状态的一些额外开销。

底线是，对于少量的可训练参数，例如在LoRA和低r（秩）值的情况下，与预训练相比，其中我们训练了更多的参数，使用SGD替换AdamW的内存收益可能非常小。

尽管SGD在这里没有提供显著的内存节省，但让我们还是快速看一下结果模型的性能：

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1703819035674-a08231e1-fa90-474f-9039-b7d7cad6bebd.png)

看来，SGD优化器的性能与AdamW相当。有趣的是，当向AdamW添加调度器时，在TruthfulQA MC2和MMLU全球事实性能上有所提高，但算术性能有所下降。（注：TruthfulQA MC2是其他公共排行榜上广为认可的基准测试。）目前，我们不会过多强调算术性能，将在本文的剩余实验中使用带调度器的AdamW。

如果您想复制这些实验，我发现最佳的AdamW学习率是3e-4，衰减率为0.01。最佳的SGD学习率是0.1，动量为0.9。在这两种情况下，我都使用了额外的100步学习率热身。

（基于这些实验，余弦调度器已被添加到Lit-GPT中，并且现在默认启用。）

### 多次迭代数据集
到目前为止，我已经用50k次迭代训练了所有模型——Alpaca数据集有50k个训练示例。一个明显的问题是，我们是否可以通过多次迭代训练集来提高模型性能，所以我用100k次迭代运行了之前的实验，这是两倍的增加：

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1703819035830-17f7c75e-4c94-4fcb-ac72-6dee3cc5266d.png)

有趣的是，增加的迭代次数导致了整体性能的下降。下降最显著的是算术基准测试。我的假设是，Alpaca数据集不包含任何相关的算术任务，当模型更多地关注其他任务时，它会主动忘记基本的算术运算。

不管怎样，如果我说这个结果不令人欣慰，那是撒谎。这样一来，我可以在本文的剩余部分继续进行较短的50k次迭代实验。

## LoRA超参数调整第一部分：对所有层启用LoRA
既然我们已经探索了围绕LoRA微调脚本的基本设置，现在让我们关注LoRA超参数本身。默认情况下，LoRA只针对多头自注意力块中的Key和Query矩阵启用。现在，我们还将其用于Value矩阵、投影层和线性层：

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1703819037135-d02b4b2a-b8d6-40f6-a7e8-159a4b0d1d29.png)

## LoRA超参数调整第二部分：增加R
LoRA参数中最重要的一个是“r”，它决定了LoRA矩阵的秩或维度，直接影响模型的复杂度和容量。较高的“r”意味着更强的表达能力，但可能导致过拟合，而较低的“r”可以减少过拟合，但代价是表达能力的降低。保持对所有层启用LoRA，我们将r从8增加到16，看看这对性能有什么影响：

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1703819037097-8099f9ed-3563-4d93-8dab-f4987b9ac6a1.png)

我们可以看到，仅仅增加r本身使结果变差了，那么发生了什么呢？让我们在下一节中找出答案。

## LoRA超参数调整第三部分：改变Alpha
在上一节中，我们增加了矩阵秩r，而保持LoRA的alpha参数不变。较高的“alpha”将更多地强调低秩结构或正则化，而较低的“alpha”将减少其影响，使模型更多地依赖原始参数。调整“alpha”有助于在拟合数据和通过正则化模型来防止过拟合之间找到平衡。

作为一个经验法则，微调LLM时通常选择一个alpha，其大小是秩的两倍（注意，这在处理扩散模型时有所不同）。让我们尝试一下，看看将alpha增加一倍会发生什么：

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1703819037394-2985b20e-2888-4050-a81b-74b9828e4090.png)

我们可以看到，将alpha增加到32现在产生了迄今为止最好的模型！但是我们又以更多的可训练参数为代价获得了这一改进：

r=8：

+ 可训练参数数量：20,277,248
+ 不可训练参数数量：6,738,415,616
+ 内存使用量：16.42 GB

r=16：

+ 可训练参数数量：40,554,496
+ 不可训练参数数量：6,738,415,616
+ 内存使用量：16.47 GB

然而，可训练参数的数量仍然足够小，以至于不会明显影响峰值内存需求。

无论如何，我们现在终于开始取得一些成果，通过更明显的幅度改进模型性能。那么，让我们继续前进，看看通过增加秩和alpha能够达到多远：

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1703819038635-8a6885fa-655d-47f0-8de9-f08fc58a4207.png)

我还进行了一些使用异常大的秩（512、1024和2048）的额外实验，但这些实验的结果较差。有些运行甚至在训练期间没有收敛到接近零的损失，这就是为什么我没有将它们添加到表格中。

到目前为止，我们可以注意到最后一行的r=256和alpha=512模型在总体上表现最佳。作为额外的对照实验，我重复了使用alpha为1的运行，并注意到大的alpha值对于良好的性能确实是必要的：

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1703819039753-1fb3cecf-c177-4cba-8e71-64b8c7d2c9a3.png)

我还重复了使用alpha值为16和32的实验，我观察到与选择alpha值为秩的两倍相比，性能同样更差。

## LoRA超参数调整第四部分：非常大的R
对于本文的最后一个调整实验，我想进一步优化上一节中最佳模型的alpha值（r=256，最后一行），怀疑它可能有点过大。

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1703819039835-7609ad1f-e0de-4105-87b7-22f30987ee47.png)

正如上表所示，当增加秩时，选择较大的alpha值似乎是至关重要的。

对于r=256和a=512的QLoRA模型，很明显我们的模型相比基础模型有了显著的改进。唯一的区域是微调模型与基础模型相比在四位数算术上的表现不足。然而，考虑到Alpaca数据集可能没有包含这样的训练示例，这是可以理解的。

上面我们看到，选择alpha为秩的两倍（例如，r=256和alpha=512）的常见建议确实产生了最佳结果，较小的alpha值导致了更差的结果。但是，将alpha增加到“秩的两倍”建议之外会怎样呢？

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1703819039261-e136165a-2b72-43b7-b39f-a293d2aa5337.png)

根据上表提供的结果，选择alpha值超过“秩的两倍”建议也使基准测试结果变差。

## 结论
本文探索了使用LoRA训练自定义LLM时可以调整的各种设置。我们发现QLoRA是一个很好的内存节省器，尽管它增加了运行时间成本。此外，尽管学习率调度器可能有益，但在AdamW和SGD优化器之间选择影响不大。而且，多次迭代数据集甚至可能使结果更糟。通过优化LoRA设置（包括秩）可以获得最佳性价比。增加秩将导致更多的可训练参数，可能导致更高程度的过拟合和运行成本。然而，增加秩时选择合适的alpha值很重要。

> 本文为国外好文翻译，原文参见链接：[链接](https://lightning.ai/pages/community/lora-insights/)
>

