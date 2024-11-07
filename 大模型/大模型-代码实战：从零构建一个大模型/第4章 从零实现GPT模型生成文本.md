# 第4章 从零实现GPT模型生成文本

本章将涵盖以下内容:

- 实现一个类似GPT的大语言模型(LLM),用于生成类人文本
- 使用层归一化来稳定神经网络训练
- 在深度神经网络中添加快捷连接以更有效地训练模型
- 实现Transformer块来创建不同规模的GPT模型
- 计算GPT模型的参数数量和存储需求

在上一章中,我们学习并实现了多头注意力机制,这是LLM的核心组件之一。在本章中,我们将实现LLM的其他构建模块,并将它们组装成一个GPT类模型,在下一章中我们将训练该模型以生成类人文本,如图4.1所示。

![Untitled](Images/大模型-从零构建一个大模型/第四章/Untitled.png)

图4.1 LLM的三个主要阶段:编码LLM架构、在通用数据集上预训练、在标记数据集上微调。本章重点关注实现LLM架构,我们将在下一章训练该架构。

图4.1中提到的LLM架构由我们将在本章实现的几个构建模块组成。我们将从模型架构的自顶向下视图开始,然后详细介绍各个组件。

## 4.1 实现LLM架构

LLM,如GPT(Generative Pretrained Transformer的缩写),是设计用于一次生成一个单词(或token)的大型深度神经网络架构。然而,尽管它们规模庞大,但模型架构并不像你可能想象的那么复杂,因为它的许多组件都是重复的,我们稍后会看到。图4.2提供了GPT类LLM的自顶向下视图,突出显示了其主要组件。

![Untitled](Images/大模型-从零构建一个大模型/第四章/Untitled1.png)

图4.2 GPT模型的示意图。除了嵌入层外,它还包含一个或多个Transformer块,其中包含我们在上一章实现的掩蔽多头注意力模块。

如图4.2所示,我们将从简单的处理开始,如输入标记化和嵌入,然后是我们的掩蔽多头注意力模块。本章的重点将放在实现GPT模型的其余结构,包括Transformer块,我们将在下一章中训练它们以生成类人文本。

在前一章中,我们处理了较小的嵌入维度以简化起见,确保我们的概念和示例可以舒适地适应单个图。现在,在本章中,我们将扩展到真实的小型GPT-2模型,具体来说是124百万参数的最小版本,如Radford等人的论文"Language Models are Unsupervised Multitask Learners"中所述。注意,虽然原始论文提到117百万参数,但这后来被修正。

第6章将重点关于加载预训练权重到这个实现中,并将其扩展到更大的GPT-2模型,具有345、762和1,542百万参数。在神经网络和LLM的上下文中,术语"参数"指的是模型的可学习权重。这些权重本质上是在训练过程中调整和优化以最小化特定损失函数的模型内部变量。这种优化允许模型从训练数据中学习。

例如,在由2,048×2,048维矩阵(或张量)表示的神经网络层中,该矩阵的每个元素都是一个参数。由于有2,048行和2,048列,该层中的总参数数量是2,048乘以2,048,等于4,194,304个参数。

> GPT-2 VS GPT-3
> 
> 
> 我们专注于GPT-2,因为OpenAI已经公开发布了预训练模型权重,我们将在第6章的实现中使用。GPT-3在模型架构方面基本相同,只是从GPT-2的1.5亿参数扩展到了GPT-3的1750亿参数,并在更多数据上进行了训练。截至本文撰写时,GPT-3的权重尚未公开可用。GPT-2也是学习实现LLM的更好选择,因为它可以在单个笔记本电脑上运行,而GPT-3需要一个GPU集群进行训练和推理。根据Lambda Labs的数据,在单个A100加速的GPU上训练GPT-3需要355年,在消费级RTX 3090 GPU上需要665年。
> 

要指定我们的小型GPT-2模型的配置,我们将使用以下Python字典,我们将在后面的示例中使用:

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # 词汇表大小
    "context_length": 1024,      # 上下文长度
    "emb_dim": 768,       # 嵌入维度
    "n_heads": 12,        # 注意力头数量
    "n_layers": 12,       # 层数
    "drop_rate": 0.1,     # Dropout率
    "qkv_bias": False     # Query-Key-Value偏置
}

```

在GPT_CONFIG_124M字典中,我们指定了几个关键参数:

- "vocab_size"指的是一个50,257个单词的词汇表,由第2章中的BPE分词器给出。
- "context_length"表示模型可以处理的最大输入标记数,用于第2章讨论的位置嵌入。
- "emb_dim"表示嵌入大小,将每个标记转换为768维向量。
- "n_heads"表示第3章实现的多头注意力机制中的注意力头数量。
- "n_layers"指定模型中Transformer块的数量,我们将在接下来的部分详细介绍。
- "drop_rate"表示dropout机制的强度(0.1意味着10%的隐藏单元被丢弃)以防止过拟合,如第3章所述。
- "qkv_bias"决定是否在多头注意力的Linear层中包含偏置向量用于查询、键和值计算。我们最初会禁用这个,遵循现代LLM的规范,但会在第6章将预训练GPT-2权重从HuggingFace加载到我们的模型时重新访问它。

给定上面的配置,我们将通过在本节中实现GPT骨架架构(DummyGPTModel)开始本章,如图4.3所示。这将为我们提供模型如何组合在一起的高级视图,以及我们需要在接下来的部分中实现哪些其他组件来组装完整的GPT模型架构。

![Untitled](Images/大模型-从零构建一个大模型/第四章/Untitled2.png)

图4.3 概述我们编码GPT架构的顺序的示意模型。在本章中,我们将从GPT骨架开始,这是一个占位符架构,然后我们将讨论各个核心部分,最终在Transformer块中组装它们以形成最终的GPT架构。

图4.3中显示的编号框说明了我们处理构建最终GPT架构所需的各个概念的顺序。我们将从步骤1开始,这是一个我们称为DummyGPTModel的GPT骨架:

```python
import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x

```

DummyGPTModel类在这里定义了一个简化版本的GPT类模型,使用PyTorch的神经网络模块(nn.Module)。该模型架构在DummyGPTModel类中包括标记和位置嵌入、dropout、一系列Transformer块(DummyTransformerBlock)、最终层归一化(DummyLayerNorm)和线性输出层(out_head)。配置作为Python字典传入,例如我们之前创建的GPT_CONFIG_124M字典。

forward方法描述了数据如何通过模型流动:它计算标记和位置嵌入、添加dropout、通过Transformer块处理数据、应用归一化,最后通过线性输出层产生logits。

上面的代码已经是可以运行的,因为我们稍后会看到,在本节之后我们准备好输入数据。然而,现在请注意上面代码中我们编码了占位符(DummyLayerNorm和DummyTransformerBlock)用于Transformer块和层归一化,我们将在后面的章节中开发。

现在,我们将准备输入数据并初始化一个新的GPT模型来说明其用法。基于我们在第2章中编码的分词器,图4.4提供了高级概述,说明数据如何流经GPT模型。

![Untitled](Images/大模型-从零构建一个大模型/第四章/Untitled3.png)

图4.4 大致概述了输入数据如何被标记化、嵌入并输入GPT模型。注意,在我们之前编码的DummyGPTClass中,标记嵌入是在GPT模型内部处理的。在LLM中,嵌入的输入标记维度通常与输出维度相匹配。这里的输出嵌入代表我们在第3章讨论的上下文向量。

为了实现图4.4中所示的步骤,我们使用第2章介绍的分词器对GPT模型的两个文本输入进行分词,创建一个批次:

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

```

两个文本的结果标记ID如下:

```
tensor([[ 6109,  3626,  6100,   345],
        [ 6109,  1110,  6622,   257]])

```

现在,我们初始化一个新的124百万参数DummyGPTModel实例并将分词后的批次输入其中:

```python
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)

```

模型输出,通常称为logits,如下所示:

```
Output shape: torch.Size([2, 4, 50257])
tensor([[[-1.2034,  0.3201, -0.7130,  ..., -1.5548, -0.2390, -0.4667],
         [-0.1192,  0.4539, -0.4432,  ...,  0.2392,  1.3469,  1.2430],
         [ 0.5307,  1.6720, -0.4695,  ...,  1.1966,  0.0111,  0.5835],
         [ 0.0139,  1.6755, -0.3388,  ...,  1.1586, -0.0435, -1.0400]],

        [[-1.0908,  0.1798, -0.9484,  ..., -1.6047,  0.2439, -0.4530],
         [-0.7860,  0.5581, -0.0610,  ...,  0.4835, -0.0077,  1.6621],
         [ 0.3567,  1.2698, -0.6398,  ..., -0.0162, -0.1296,  0.3717],
         [-0.2407, -0.7349, -0.5102,  ...,  2.0057, -0.3694,  0.1814]]],
       grad_fn=<UnsafeViewBackward0>)

```

输出张量有两行,对应于两个文本样本。每一行包含4个标记;每个标记是一个50,257维向量,匹配分词器词汇表的大小。

嵌入有50,257个维度,因为这些维度中的每一个都对应词汇表中的一个唯一标记。在本章结束时,当我们实现相应的解码器时,我们将把这些50,257维向量转换回标记ID,然后我们可以解码成单词。

现在我们已经对GPT架构及其输入和输出有了高层次的了解,我们将在接下来的部分中填充各个占位符,从将替换前面代码中的DummyLayerNorm的实际层归一化类开始。

## 4.2 使用层归一化标准化激活

训练具有许多层的深度神经网络有时可能具有挑战性,并导致诸如梯度消失或梯度爆炸等问题。这些问题会导致不稳定的训练动态,使网络难以有效调整其权重,这意味着学习过程难以找到一组参数(权重)来最小化损失函数。换句话说,网络难以学习数据中的底层模式,这将允许它对未见过的数据做出准确预测。(如果你不熟悉神经网络训练和梯度的概念,可以在附录A:深度学习入门的B.4节自动微分和损失中找到对这些概念的简要介绍。然而,要理解本文的内容,不需要对梯度有深入的数学理解。)

在本节中,我们将实现层归一化来提高神经网络训练的稳定性和效率。

层归一化背后的主要思想是调整神经网络层的激活(输出)使其具有0的均值和1的方差,也称为单位方差。这种调整加速了收敛到有效权重并提高了一致性,实现可靠的训练。正如我们在前面部分看到的,基于DummyLayerNorm占位符,在GPT-2和现代Transformer架构中,层归一化通常应用在多头注意力模块之前和之后,以及最终输出层之前。

在我们用代码实现层归一化之前,图4.5提供了层归一化如何工作的可视化概述。

![Untitled](Images/大模型-从零构建一个大模型/第四章/Untitled4.png)

图4.5 层归一化的示意图,其中5个层输出(也称为激活)被归一化,使其具有零均值和1的方差。

我们可以使用以下代码重现图4.5中所示的示例,其中我们实现了一个具有5个输入和6个输出的神经网络层,我们将其应用于两个输入样本:

```python
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)

```

这将输出以下张量,其中前两行是第一个输入的层输出,后两行是第二个输出的层输出:

```
tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
        [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]],
       grad_fn=<ReluBackward0>)

```

我们刚刚编码的神经网络层由一个Linear层组成,后跟一个非线性激活函数ReLU(Rectified Linear Unit的缩写),这是神经网络中的标准激活函数。如果你不熟悉ReLU,它简单地将负输入阈值为0,确保层输出只有正值,这解释了为什么结果层输出不包含任何负值。(注意,我们将在下一节介绍另一种更复杂的激活函数GELU,我们将在GPT中使用。)

在我们对这些输出应用层归一化之前,让我们检查均值和方差:

```python
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:\\n", mean)
print("Variance:\\n", var)

```

输出如下:

```
Mean:
  tensor([[0.1324],
          [0.2170]], grad_fn=<MeanBackward1>)
Variance:
  tensor([[0.0231],
          [0.0398]], grad_fn=<VarBackward0>)

```

上面均值张量中的第一行包含第一个输入行的均值,第二个输出行包含第二个输入行的均值。

在均值或方差计算操作中使用keepdim=True确保输出张量保持与输入张量相同数量的维度,即使操作减少了沿指定维度的张量。例如,没有keepdim=True,返回的均值张量将是一个2维向量[0.1324, 0.2170]而不是一个2×1维矩阵[[0.1324], [0.2170]]。

dim参数指定应该在张量的哪个维度上执行统计(如均值或方差)的计算,如图4.6所示。

![Untitled](Images/大模型-从零构建一个大模型/第四章/Untitled5.png)

图4.6 dim参数在计算张量均值时的说明。例如,如果我们有一个2D张量(矩阵)维度为[行,列],使用dim=0将在行之间执行操作(垂直,如底部所示),结果输出聚合每列的数据。使用dim=1或dim=-1将在列之间执行操作(水平,如顶部所示),结果输出聚合每行的数据。

如图4.6所示,对于2D张量(只是一个矩阵),使用dim=-1进行操作与使用dim=1进行均值或方差计算相同。这是因为-1指的是张量的最后一个维度,对应于2D张量中的列。事实上,当将层归一化应用于GPT模型时,该模型生成3D张量,形状为[batch_size, num_tokens, embedding_size],我们仍然可以使用dim=-1进行最后一个维度的归一化,避免从dim=1更改为dim=2。

现在,让我们通过减去均值并除以方差的平方根(也称为标准差)来对我们之前获得的层输出应用层归一化:

```python
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Normalized layer outputs:\\n", out_norm)
print("Mean:\\n", mean)
print("Variance:\\n", var)

```

正如我们从结果中看到的,归一化的层输出,现在包含负值,具有接近0的均值和1的方差:

```
Normalized layer outputs:
 tensor([[ 0.6159,  1.4126, -0.8719,  0.5872, -0.8719, -0.8719],
        [-0.0189,  0.1121, -1.0876,  1.5173,  0.5647, -1.0876]],
       grad_fn=<DivBackward0>)
Mean:
 tensor([[2.9802e-08],
        [3.9736e-08]], grad_fn=<MeanBackward1>)
Variance:
 tensor([[1.],
        [1.]], grad_fn=<VarBackward0>)

```

注意,输出张量中的值2.9802e-08是科学记数法表示2.9802 × 10^-8,在十进制形式中为0.0000000298。这个值非常接近0,但不完全是0,这是由于计算机表示数字时的小数值误差造成的。

为了提高可读性,我们可以通过在打印张量值时设置sci_mode为False来关闭科学记数法:

```python
torch.set_printoptions(sci_mode=False)
print("Mean:\\n", mean)
print("Variance:\\n", var)
Mean:
 tensor([[    0.0000],
        [    0.0000]], grad_fn=<MeanBackward1>)
Variance:
 tensor([[1.],
        [1.]], grad_fn=<VarBackward0>)

```

到目前为止,在本节中,我们已经逐步编码并应用了层归一化。现在让我们将这个过程封装在一个PyTorch模块中,我们可以在GPT模型中使用:

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

```

这个高效的层归一化实现在输入张量x的最后一个维度上操作,该维度表示嵌入维度(emb_dim)。变量eps是一个小常数(epsilon)添加到方差中以防止在归一化过程中除以零。scale和shift是两个可训练参数(与输入相同维度),LLM在训练期间自动调整,如果确定这样做会提高模型在其训练任务上的性能。这允许模型学习适当的缩放和偏移,使其适应正在处理的任务。

> 有偏方差
> 
> 
> 在方差计算方法中,我们注意到一个实现细节是设置unbiased=False。对于好奇的读者,在方差计算中,我们除以输入数量n而不是n-1。这种方法不应用Bessel的校正,这通常使用n-1而不是n作为分母来调整样本方差估计的偏差。这种方法导致所谓的有偏估计的方差。对于大规模语言模型(LLM),其中嵌入维度n显著较大,使用n和n-1之间的差异在实践中可以忽略不计。我们选择这种方法以确保与预训练GPT-2模型的归一化层兼容,并且因为它反映了PyTorch的默认行为,用于实现原始GPT-2模型。使用类似的设置确保我们的方法与我们将在第6章使用的预训练权重兼容。
> 

让我们现在测试LayerNorm模块并将其应用于我们的批处理输入:

```python
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("Mean:\\n", mean)
print("Variance:\\n", var)

```

正如我们可以从结果看到的,层归一化如预期工作,将两个输入批次中的每一个归一化为具有接近0的均值和1的方差:

```
Mean:
 tensor([[    -0.0000],
        [     0.0000]], grad_fn=<MeanBackward1>)
Variance:
 tensor([[1.0000],
        [1.0000]], grad_fn=<VarBackward0>)

```

在本节中,我们介绍了我们需要实现GPT架构的构建块之一,如图4.7中的心智模型所示。

![Untitled](Images/大模型-从零构建一个大模型/第四章/Untitled6.png)

图4.7 列出我们在本章实现的不同构建块的心智模型,虚线框表示我们已经完成的部分。

在下一节中,我们将转向GELU激活函数,这是LLM中使用的激活函数之一,而不是我们在本节中使用的传统ReLU函数。

## 4.3 使用GELU激活实现前馈网络

在本节中,我们将实现一个小型线性网络子模块,它被用作Transformer块中的一部分在LLM中。我们从实现GELU激活函数开始,它在这个神经网络子模块中扮演着关键角色。(有关在PyTorch中实现神经网络的其他信息,请参见附录C中的第C.5节实现多层神经网络。)

历史上,ReLU激活函数因其简单性和在各种神经网络架构中的有效性而被广泛使用。然而,在LLM中,几种其他激活函数被采用超出了传统的ReLU。两个值得注意的例子是GELU(高斯误差线性单元)和SwiGLU(Swish-门控线性单元)。

GELU和SwiGLU是更复杂和平滑的激活函数,分别结合高斯和sigmoid-门控线性单元。它们为这些大型语言模型提供了改进的性能,不像简单的ReLU。

GELU激活函数可以通过多种方式实现;确切版本定义为GELU(x)=x⋅Φ(x),其中Φ(x)是标准高斯分布的累积分布函数。然而,在实践中,通常实现一个计算上更高效的近似(原始GPT-2模型也是用这个近似训练的):

GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))

在代码中,我们可以将此函数实现为PyTorch模块如下:

```python
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

```

现在,为了对比这个GELU函数看起来如何以及它如何与ReLU函数比较,让我们绘制这些函数:

```python
import matplotlib.pyplot as plt
gelu, relu = GELU(), nn.ReLU()

x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
plt.show()

```

正如我们可以在生成的图形中看到的(图4.8),ReLU是一个分段线性函数,对于正输入直接输出输入;否则,它输出零。GELU是一个平滑的非线性函数,近似ReLU但具有非零负值梯度。

![Untitled](Images/大模型-从零构建一个大模型/第四章/Untitled7.png)

图4.8 GELU和ReLU函数的图形比较。x轴显示函数输入,y轴显示函数输出。

GELU的平滑性,如图4.8所示,可以导致在训练期间更好的优化属性,因为它允许对模型参数进行更细微的调整。相比之下,ReLU在零处有一个尖锐的拐角,这有时可能阻碍优化,特别是在需要更复杂架构的网络中。此外,不像ReLU对任何负输入输出零,GELU允许小的非零输出用于负值。这个特性意味着在训练过程中,神经元接收负输入仍然可以对学习过程做出贡献,尽管程度较小于正输入。

现在,让我们使用GELU函数来实现小型神经网络模块FeedForward,我们将在GPT的Transformer块中使用:

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

```

如我们在前面的代码中看到的,FeedForward模块是一个小型线性网络,由两个Linear层和一个GELU激活函数组成。在124百万参数GPT模型中,它接收输入批次,其中的标记具有768的嵌入大小,如GPT_CONFIG_124M字典所指定,其中GPT_CONFIG_124M["emb_dim"] = 768。

图4.9说明了当我们通过它传递输入时,嵌入大小如何在这个小型前馈神经网络内部被操纵。

![Untitled](Images/大模型-从零构建一个大模型/第四章/Untitled8.png)

图4.9 前馈神经网络中层输出的扩展和收缩的示意图。首先,输入从768扩展到3072个值。然后,第二层将3072个值压缩回768维表示。

值得注意的是,这个神经网络可以适应可变的批量大小和输入中的标记数量。然而,每个标记的嵌入大小在初始化权重时是确定的并固定的。

遵循图4.9中的示例,让我们初始化一个新的FeedForward模块,其中标记嵌入大小为768,并将其应用于一个具有2个样本和3个标记每个的批处理输入:

```python
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)

```

正如我们可以看到,输出张量的形状与输入张量的形状相同:

```
torch.Size([2, 3, 768])

```

我们实现的FeedForward模块在增强模型学习和泛化数据的能力方面扮演着关键角色。虽然这个模块的输入和输出维度相同,但它在内部将嵌入维度扩展到更高维空间,通过第一个线性层如图4.10所示。这个扩展后跟着一个非线性GELU激活,然后是一个收缩回原始维度的第二个线性变换。这样的设计允许探索更丰富的表示空间。

此外,输入和输出维度的一致性简化了架构,通过允许多个层的堆叠,我们稍后会看到,而无需在它们之间调整维度,从而使模型更具可扩展性。

如图4.10所示,我们现在已经实现了GPT的构建块中的更多部分。

![Untitled](Images/大模型-从零构建一个大模型/第四章/Untitled9.png)

图4.10 展示我们在本章中涵盖的主题的心智模型,黑色对勾表示我们已经涵盖的内容。

在下一节中,我们将探讨在不同层之间添加到神经网络的快捷连接的概念,这对于改善深度神经网络架构中的训练性能非常重要。

图 4.11 是一个心智模型，显示了我们在本章中涵盖的主题，黑色的对号表示我们已经覆盖的内容。

![Untitled](Images/大模型-从零构建一个大模型/第四章/Untitled10.png)

## 4.4 添加快捷连接

现在,让我们讨论快捷连接背后的概念,也称为残差连接。最初,快捷连接被提出用于非常深的网络在计算机视觉(特别是在残差网络中)以缓解梯度消失的挑战。梯度消失问题指的是梯度(指导权重更新during训练)在它们通过层传播反向时变得越来越小的问题,使得有效训练较早的层变得困难,如图4.12所示。

![Untitled](Images/大模型-从零构建一个大模型/第四章/Untitled11.png)

图4.12 对比由5层组成的深度神经网络,左侧没有快捷连接,右侧有快捷连接。快捷连接涉及将一层的输入添加到其输出,有效地创建一条绕过某些层的替代路径。图中的梯度说明表示每层的平均绝对梯度,我们将在接下来的代码示例中计算。

如图4.12所示,快捷连接通过跳过一个或多个层将一层的输出直接馈送到更深层,从而创建一条替代的、更短的路径供梯度流过网络。这就是为什么这些连接也被称为残差连接。它们在训练期间反向传播过程中保护梯度流发挥着关键作用。

在下面的示例中,我们实现图4.12中所示的神经网络,以展示如何在forward方法中使用快捷连接:

```python
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            # 实现5个层
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            # 计算当前层的输出
            layer_output = layer(x)
            # 检查是否可以应用快捷连接
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

```

这段代码实现了一个深度神经网络,有5个层,每个都由一个Linear层和一个GELU激活函数组成。在forward方法中,我们迭代地将输入通过层,并有选择地应用图4.12中描述的快捷连接,如果self.use_shortcut属性被设置为True。

让我们使用这个代码首先初始化一个没有快捷连接的神经网络。这里,每个层将被初始化,使其接受一个具有3个输入值的示例并返回3个输出值。最后一层返回单个输出值:

```python
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123) # 指定随机种子以确保初始权重的可重复性
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)

```

现在,我们实现一个函数,计算模型在反向传播过程中的梯度:

```python
def print_gradients(model, x):
    # 前向传播
    output = model(x)
    target = torch.tensor([[0.]])

    # 计算损失,基于目标和输出的接近程度
    loss = nn.MSELoss()
    loss = loss(output, target)

    # 反向传播计算梯度
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            # 打印权重的平均绝对梯度
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

```

在上面的代码中,我们指定一个损失函数,计算模型输出和一个预定义目标(这里为简单起见,值为0)之间的距离。然后,当调用loss.backward()时,PyTorch计算每层模型中所有参数的梯度。我们可以通过model.named_parameters()迭代权重参数。假设我们有一个3×3的权重参数矩阵用于给定层。在这种情况下,每层将有3×3个梯度值,我们打印这些3×3个梯度值的平均绝对梯度,以获得每层的单个梯度值来比较层之间的梯度。

简而言之,.backward()方法是PyTorch中的一个便捷方法,用于计算所有梯度,这在模型训练期间是必需的,而无需自己实现梯度计算的数学,从而使使用深度神经网络变得更加容易。如果你不熟悉梯度和神经网络训练的概念,我建议阅读附录A中的B.4节,自动微分和损失以及C.7节典型训练循环。

让我们现在使用print_gradients函数并将其应用于没有快捷连接的模型:

```python
print_gradients(model_without_shortcut, sample_input)

```

输出如下:

```
layers.0.0.weight has gradient mean of 0.00020173587836325169
layers.1.0.weight has gradient mean of 0.0001201116101583466
layers.2.0.weight has gradient mean of 0.0007152041653171182
layers.3.0.weight has gradient mean of 0.001398873864673078
layers.4.0.weight has gradient mean of 0.005049646366387606

```

正如我们可以从print_gradients函数的输出中看到,梯度在从最后一层(layers.4)到第一层(layers.0)的传播过程中变得越来越小,这是梯度消失问题的一个表现。

让我们现在初始化一个带有快捷连接的模型,看看它如何比较:

```python
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)
print_gradients(model_with_shortcut, sample_input)

```

输出如下:

```
layers.0.0.weight has gradient mean of 0.22169792652130127
layers.1.0.weight has gradient mean of 0.20694105327129364
layers.2.0.weight has gradient mean of 0.32896995544433594
layers.3.0.weight has gradient mean of 0.2665732502937317
layers.4.0.weight has gradient mean of 1.3258541822433472

```

正如我们可以看到,根据输出,最后一层(layers.4)仍然有较大的梯度值。然而,梯度值在向第一层(layers.0)传播时保持稳定,并没有缩小到消失的小值。

总之,快捷连接对于克服由梯度消失问题引起的优化挑战在深度神经网络中非常重要。快捷连接是像LLM这样的大型模型的一个核心构建块,当我们在下一章训练GPT模型时,它们将有助于更有效的训练,确保跨层的一致梯度流。

介绍了快捷连接后,我们现在将在下一节中将所有先前涵盖的概念(层归一化、GELU激活、前馈模块和快捷连接)连接在一个Transformer块中,这是我们需要完成GPT架构的最后一个构建块。

## 4.5 在Transformer块中连接注意力和线性层

在本节中,我们将实现Transformer块,这是GPT和其他LLM架构的一个基本构建块。这个块,在124百万参数GPT-2架构中重复12次,结合了我们之前涵盖的几个概念:多头注意力、层归一化、dropout、前馈层和GELU激活,如图4.13所示。在下一节中,我们将把这个Transformer块连接到GPT架构的其余部分。

![Untitled](Images/大模型-从零构建一个大模型/第四章/Untitled12.png)

图4.13 Transformer块的示意图。图的底部显示已嵌入到768维向量的输入标记。每行对应一个标记的向量表示。Transformer块的输出是与输入相同维度的向量,然后可以作为LLM中后续层的输入。

如图4.13所示,Transformer块结合了几个组件,包括第3章中的掩蔽多头注意力模块和我们在4.3节实现的FeedForward模块。

当Transformer块处理输入序列时,序列中的每个元素(例如,一个单词或子词标记)都由固定大小的向量表示(在图4.13的情况下,768维)。Transformer块内的操作,包括多头注意力和前馈层,都被设计为转换这些向量,同时保持它们的维度。

关键是注意力机制在多头注意力块中识别和分析输入序列中元素之间的关系。相比之下,前馈网络独立修改每个位置的特征。这种组合既允许对输入的更丰富理解和处理,又增强了模型处理复杂数据模式的整体能力。

在代码中,我们可以如下创建TransformerBlock:

```python
from previous_chapters import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 加回原始输入

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

```

上面的代码定义了一个TransformerBlock类在PyTorch中,它包括一个多头注意力机制(MultiHeadAttention)和一个前馈网络(FeedForward),根据提供的配置字典(cfg)进行配置,如GPT_CONFIG_124M。

层归一化(LayerNorm)在这两个组件之前应用,dropout在之后应用以规范化模型并防止过拟合。这被称为Pre-LayerNorm。其他架构,如原始Transformer模型,在多头注意力和前馈网络之后应用层归一化,称为Post-LayerNorm,这通常会导致更差的训练动态。

该类还实现了前向传递,其中每个组件后面都跟着一个快捷连接,将块的输入添加到其输出。这个关键特性帮助梯度在训练期间流过网络,并改善深度模型的学习,如4.4节所解释。

使用我们之前定义的GPT_CONFIG_124M字典,让我们实例化一个Transformer块并将其应用于一些样本数据:

```python
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

```

输出如下:

```
Input shape: torch.Size([2, 4, 768])
Output shape: torch.Size([2, 4, 768])

```

正如我们可以从上面的输出看到,Transformer块在其输出中保持了输入维度,表明Transformer架构处理序列而不改变它们在网络中的形状。

输入和输出形状在整个Transformer块架构中的保持不变不是偶然的,而是其设计的一个关键方面。这种设计使其能够有效地应用于各种序列到序列任务,其中每个输出向量直接对应一个输入向量,维持一对一的关系。然而,输出是一个上下文向量,它封装了来自整个输入序列的信息,正如我们在第3章中讨论的。这意味着虽然序列的物理维度(长度和特征大小)在通过Transformer块时保持不变,但每个输出向量的内容被重新编码以整合来自整个输入序列的上下文信息。

有了我们在本节中实现的Transformer块,我们现在拥有了所有构建块,如图4.14所示,需要在下一节中实现GPT架构。

![Untitled](Images/大模型-从零构建一个大模型/第四章/Untitled13.png)

图4.14 我们在本章到目前为止实现的不同概念的心智模型。

如图4.14所示,Transformer块结合了层归一化、前馈网络(包括GELU激活)和快捷连接,这些我们在本章前面已经介绍过。正如我们将在下一章看到的,这个Transformer块将成为我们将实现的GPT架构的主要组成部分。

## 4.6 编码GPT模型

我们开始本章时有一个GPT架构的高级概述,我们称之为DummyGPTModel。在那个DummyGPTModel实现中,我们展示了GPT模型的输入和输出,但其构建块仍然是黑盒,使用DummyTransformerBlock和DummyLayerNorm类作为占位符。

在本节中,我们现在将用我们在本章后面编码的实际TransformerBlock和LayerNorm类替换DummyTransformerBlock和DummyLayerNorm占位符,以组装原始124百万参数版本GPT-2的完全工作版本。在第5章中,我们将训练一个GPT-2模型,在第6章中,我们将加载预训练权重从HuggingFace。

在我们用代码组装GPT-2模型之前,让我们回顾一下图4.15中的整体结构,它结合了我们在本章中涵盖的所有概念。

![Untitled](Images/大模型-从零构建一个大模型/第四章/Untitled14.png)

图4.15 GPT模型架构概述。这个图展示了数据如何通过GPT模型流动。从底部开始,tokenized文本首先被转换为token嵌入,然后与位置嵌入结合。这个组合信息形成一个张量,该张量通过中间显示的一系列Transformer块(每个包含多头注意力和前馈神经网络层,带有dropout和层归一化),这些块堆叠在一起并重复12次。

如图4.15所示,我们在4.5节中编码的Transformer块在GPT模型架构中重复多次。在124百万参数GPT-2模型的情况下,它重复12次,这是我们在GPT_CONFIG_124M字典中通过"n_layers"条目指定的。对于最大的GPT-2模型,有1,542百万参数,这个Transformer块重复36次。

如图4.15所示,最终Transformer块的输出然后通过一个最终层归一化,然后再到达线性输出层。这一层使用Transformer的输出到高维空间(在这种情况下,50,257维,对应于模型的词汇表大小)来预测序列中的下一个token。

现在让我们用代码实现我们在图4.15中看到的架构:

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

```

由于我们在4.5节中实现的TransformerBlock类,GPTModel类相对较小和紧凑。

这个GPTModel类的__init__构造函数使用作为Python字典传入的配置初始化token和位置嵌入层。这些嵌入层负责将输入token索引转换为密集向量并添加位置信息,如第2章中讨论的。

然后,__init__方法创建一个TransformerBlock模块的顺序堆栈,等于cfg中指定的层数。在Transformer块之后,应用一个LayerNorm层,标准化来自Transformer块的输出以稳定学习过程。最后,定义一个没有偏置的线性输出头,将Transformer的输出投影到分词器的词汇空间以生成每个词汇表token的logits。

forward方法接受一批输入token索引,计算它们的嵌入,应用位置嵌入,通过Transformer块传递序列,规范化最终输出,然后计算logits,代表词汇表中每个token的未归一化概率。我们将在下一节中将这些logits转换为token和文本输出。

现在让我们使用GPT_CONFIG_124M字典初始化124百万参数GPT模型,并将我们在本章开始时创建的批处理输入传递给它:

```python
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\\n", batch)
print("\\nOutput shape:", out.shape)
print(out)

```

上面的代码打印输入批次的内容,后跟输出张量:

```
Input batch:
 tensor([[ 6109,  3626,  6100,   345], # 文本1的token ID
         [ 6109,  1110,  6622,   257]]) # 文本2的token ID

Output shape: torch.Size([2, 4, 50257])
tensor([[[ 0.3613,  0.4222, -0.0711,  ...,  0.3483,  0.4661, -0.2838],
         [-0.1792, -0.5660, -0.9485,  ...,  0.0477,  0.5181, -0.3168],
         [ 0.7120,  0.0332,  0.1085,  ...,  0.1018, -0.4327, -0.2553],
         [-1.0076,  0.3418, -0.1190,  ...,  0.7195,  0.4023,  0.0532]],

        [[-0.2564,  0.0900,  0.0335,  ...,  0.2659,  0.4454, -0.6806],
         [ 0.1230,  0.3653, -0.2074,  ...,  0.7705,  0.2710,  0.2246],
         [ 1.0558,  1.0318, -0.2800,  ...,  0.6936,  0.3205, -0.3178],
         [-0.1565,  0.3926,  0.3288,  ...,  1.2630, -0.1858,  0.0388]]],
       grad_fn=<UnsafeViewBackward0>)

```

正如我们可以看到,输出张量具有形状[2, 4, 50257],因为我们传入了2个输入文本,每个有4个token。最后一个维度,50257,对应于分词器的词汇表大小。在下一节中,我们将看到如何将这些50,257维输出向量转换回token。

在我们继续下一节并编写将模型输出转换为文本的函数之前,让我们花点时间更深入地研究模型架构本身并分析其数据。

使用numel()方法,意思是"元素数量",我们可以计算模型参数张量中参数的总数:

```python
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

```

结果如下:

```
Total number of parameters: 163,009,536

```

现在,一个好奇的读者可能会注意到一个矛盾。早些时候,我们谈到初始化一个124百万参数GPT模型,那么为什么实际参数数量是163百万,如上面的输出所示?

原因是一个叫做权重绑定的概念,它在原始GPT-2架构中使用,这意味着原始GPT-2架构正在重用token嵌入层的权重作为其输出层。为了理解这意味着什么,让我们看看token嵌入层和线性输出层的形状,我们在model中为GPTModel初始化:

```python
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)

```

正如我们可以从打印输出看到的,这两层的权重张量具有相同的形状:

```
Token embedding layer shape: torch.Size([50257, 768])
Output layer shape: torch.Size([50257, 768])

```

token嵌入和输出层是如此之大,因为50,257对应于分词器的词汇表中的单词数量。让我们从总GPT-2模型计数中移除输出层参数计数,按照权重绑定:

```python
total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

```

输出如下:

```
Number of trainable parameters considering weight tying: 124,412,160

```

正如我们可以看到,模型现在恰好有124百万参数,匹配原始GPT-2模型的大小。

权重绑定减少了模型的整体内存占用和计算复杂度。然而,根据我的经验,使用单独的token嵌入和输出层会导致更好的训练和模型性能;因此,我们使用单独的层在GPTModel实现中。这种方法更常见于现代LLM。然而,我们将在第6章中重新访问并实现权重绑定概念,当我们加载预训练权重从HuggingFace时。

> 练习4.1 前馈和注意力模块中的参数数量
> 
> 
> 计算并比较包含在前馈模块中的参数数量与包含在多头注意力模块中的参数数量。
> 

最后,让我们计算163百万参数在GPTModel对象中的内存需求:

```python
total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")

```

结果如下:

```
Total size of the model: 621.83 MB

```

总结一下,通过计算GPTModel对象中163百万参数的内存需求并假设每个参数是32位浮点数占4字节,我们发现模型的总大小约为621.83 MB,说明即使相对较小的LLM也需要相当大的存储容量。

在本节中,我们实现了GPTModel架构,并看到它输出数值张量,形状为[batch_size, num_tokens, vocab_size]。在下一节中,我们将编写代码将这些输出张量转换为文本。

> 练习4.2 初始化更大的GPT模型
> 
> 
> 在本章中,我们初始化了一个124百万参数的GPT模型,称为"GPT-2 small"。不做任何代码修改,只需更新配置文件,你能使用GPTModel类来实现GPT-2 medium(使用1024维嵌入,24个Transformer块,16个多头注意力头),GPT-2 large(1280维嵌入,36个Transformer块,20个多头注意力头),和GPT-2 XL(1600维嵌入,48个Transformer块,25个多头注意力头)吗?作为额外练习,计算每个GPT模型的总参数数量。
> 

## 4.7 生成文本

在本章的最后一节,我们将实现代码,将GPT模型的张量输出转换为文本。在我们开始之前,让我们简要回顾一下生成模型如何在LLM中一次生成一个token(或单词)的文本,如图4.16所示。

![Untitled](Images/大模型-从零构建一个大模型/第四章/Untitled15.png)

图4.16 这个图说明了LLM生成文本的逐步过程,一次一个token。从初始输入上下文("Hello, I am")开始,模型在每次迭代中预测后续token,将其附加到下一轮预测的输入上下文中。如图所示,第一次迭代添加"a",第二次"model",第三次"ready",逐步构建句子。

图4.16说明了LLM模型如何在高层次上给定输入上下文(如"Hello, I am,")生成文本。每次迭代,输入上下文都会增长,允许模型生成连贯和上下文适当的文本。到第6次迭代,模型已经构建了一个完整的句子:"Hello, I am a model ready to help."

在上一节中,我们看到我们当前的GPTModel实现输出形状为[batch_size, num_token, vocab_size]的张量。那么,GPT模型如何使用这些输出张量来生成图4.16中所示的文本呢?

GPT模型使用其输出张量生成文本的过程涉及几个步骤,如图4.17所示。这些步骤包括解码输出张量,基于概率分布选择token,并将这些token转换为人类可读的文本。

![Untitled](Images/大模型-从零构建一个大模型/第四章/Untitled16.png)

图4.17 详细说明了GPT模型中文本生成的机制,展示了token生成过程中的单个迭代。该过程首先将输入文本编码为token ID,然后将其输入GPT模型。模型的输出随后被转换回文本并附加到原始输入文本中。

图4.17中详细的单token生成过程说明了GPT模型如何给定其输入生成下一个token。

在代码中,模型输出一个矩阵,其向量表示潜在的下一个token。对应于下一个token的向量被提取并通过softmax函数转换为概率分布。在这个概率向量中,具有最高值的索引被选择,这转化为token ID。这个token ID然后被解码回文本,生成序列中的下一个token。最后,这个token被附加到之前的输入,形成下一次迭代的新输入序列。这个逐步的过程使模型能够顺序生成文本,构建连贯的短语和句子从初始输入上下文。

我们重复这个过程多次迭代,如前面图4.16所示,直到我们达到用户指定的生成token数量。

在代码中,我们可以按如下方式实现token生成过程:

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

```

在上面的代码中,generate_text_simple函数中,我们使用softmax函数将logits转换为概率分布,从中我们用torch.argmax标识具有最高值的位置。softmax函数是单调的,意味着它保留其输入的顺序。因此,在实践中,对softmax输出使用argmax是多余的,因为logits张量中得分最高的位置与softmax输出张量中相同的位置。换句话说,我们可以直接对logits张量应用torch.argmax函数并得到相同的结果。然而,我们编码了转换以说明将logits转换为概率的完整过程,这有其他用途,例如当模型生成最可能的下一个token,这被称为贪婪解码。

在下一章,当我们实现GPT训练循环时,我们还将讨论额外的采样技术,我们修改softmax输出,使模型不总是选择最可能的token,这引入了多样性和创造性到生成的文本中。

此过程使用generate_text_simple函数一次生成一个token ID并将其附加到上下文中,如图4.18进一步说明。(单次迭代的token ID生成过程在图4.17中详细说明。)

![Untitled](Images/大模型-从零构建一个大模型/第四章/Untitled17.png)

图4.18 说明六次token预测循环迭代,其中模型以初始token ID序列作为输入,预测下一个token,并将此token附加到下一次迭代的输入序列。(token ID也被翻译成相应的文本以便更好理解。)

如图4.18所示,我们以迭代方式生成token ID。例如,在迭代1中,模型提供了对应于"Hello , I am"的token,预测下一个token(ID为257,对应于"a"),并将其附加到输入。此过程重复直到模型在六次迭代后产生完整句子"Hello, I am a model ready to help."

现在让我们使用"Hello, I am"上下文作为模型输入尝试generate_text_simple函数,如图4.18所示。

首先,我们将输入上下文编码为token ID:

```python
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

```

编码后的ID如下:

```
encoded: [15496, 11, 314, 716]
encoded_tensor.shape: torch.Size([1, 4])

```

现在,我们将模型置于.eval()模式,这禁用了在训练期间使用的随机组件如dropout,并在编码的输入张量上使用generate_text_simple函数:

```python
model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output:", out)
print("Output length:", len(out[0]))

```

结果输出token ID如下:

```
Output: tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267]])
Output length: 10

```

使用分词器的.decode方法,我们可以将ID转换回文本:

```python
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)

```

模型以文本格式的输出如下:

```
Hello, I am Featureiman Byeswickattribute argue

```

正如我们可以看到,基于上面的输出,模型生成了无意义的文本,这与图4.18中显示的连贯文本相去甚远。为什么会这样?原因是我们还没有训练模型。到目前为止,我们已经实现了GPT架构并初始化了一个GPT模型实例,其初始权重是随机的。

虽然训练本身是一个很大的话题,我们将在下一章讨论。

> 练习4.3 使用单独的Dropout参数
> 
> 
> 在本章开始时,我们在GPT_CONFIG_124M字典中定义了一个全局的"drop_rate"设置,用于整个GPTModel架构中的dropout率。修改代码以指定单独的dropout值用于模型架构中的各种dropout层。(提示:有三个不同的地方我们使用dropout层:嵌入层、快捷层和多头注意力模块。)
> 

## 4.8 总结

- 层归一化通过确保每层输出具有一致的均值和方差来稳定训练。
- 快捷连接是跳过一个或多个层的连接,通过将一层的输出直接馈送到更深层,这有助于缓解训练深度神经网络(如LLM)时的梯度消失问题。
- Transformer块是GPT模型的核心结构组件,结合了掩蔽多头注意力模块和使用GELU激活函数的全连接前馈网络。
- GPT模型是具有许多重复Transformer块的LLM,具有数百万到数十亿个参数。
- GPT模型有不同的大小,例如124、345、762和1542百万参数,我们可以用相同的GPTModel Python类实现。
- GPT类LLM的文本生成能力涉及将输出张量解码为人类可读的文本,通过基于给定输入上下文顺序预测一个token来实现。
- 没有训练,GPT模型生成不连贯的文本,这突出了模型训练对于连贯文本生成的重要性,这是后续章节的主题。