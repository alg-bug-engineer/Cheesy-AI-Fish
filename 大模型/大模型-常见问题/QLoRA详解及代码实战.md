### 背景
<font style="color:#8A8F8D;">论文标题:QLoRA: Efficient Finetuning of Quantized LLMs</font>

<font style="color:#8A8F8D;">论文链接:https://arxiv.org/pdf/2305.14314.pdf</font>

**<font style="color:rgb(25, 27, 31);background-color:#F9EFCD;">QLoRA</font>**<font style="color:rgb(25, 27, 31);background-color:#F9EFCD;">是当前PEFT利器，将</font><font style="color:#601BDE;background-color:#F9EFCD;">预训练模型</font><font style="color:rgb(25, 27, 31);background-color:#F9EFCD;">的参数 </font><font style="color:rgb(25, 27, 31);background-color:#F9EFCD;">W</font><font style="color:rgb(25, 27, 31);background-color:#F9EFCD;">量化到</font><font style="color:rgba(252,7,20,1);background-color:#F9EFCD;">NF4</font><font style="color:rgb(25, 27, 31);background-color:#F9EFCD;">的精度（</font><font style="color:rgba(252,7,20,1);background-color:#F9EFCD;">存储数据</font><font style="color:rgb(25, 27, 31);background-color:#F9EFCD;">），在进行特征计算时，通过双重反量化将它还原到</font><font style="color:rgba(252,7,20,1);background-color:#F9EFCD;">BF16</font><font style="color:rgb(25, 27, 31);background-color:#F9EFCD;">精度</font>**<font style="color:#601BDE;background-color:#F9EFCD;">参数更新</font>**<font style="color:rgb(25, 27, 31);background-color:#F9EFCD;">（</font><font style="color:rgba(252,7,20,1);background-color:#F9EFCD;">计算数据类型</font><font style="color:rgb(25, 27, 31);background-color:#F9EFCD;">）</font><font style="color:rgb(25, 27, 31);">。同LoRA一样，QLoRA也</font><font style="color:#601BDE;">在原参数一侧添加了一个与原参数并行的</font>**<font style="color:#601BDE;">低秩适配器</font>**<font style="color:rgb(25, 27, 31);">，QLoRA能在<8GB显存GPU Fintune LLaMA2-7B。</font>

### <font style="color:rgb(25, 27, 31);">创新点</font>
<font style="color:rgba(0, 0, 0, 0.9);">QLoRA将</font><font style="color:#601BDE;">低精度存储</font><font style="color:rgba(0, 0, 0, 0.9);">（NF4）与</font><font style="color:#601BDE;">高精度计算</font><font style="color:rgba(0, 0, 0, 0.9);">（BFloat16）结合起来，</font>**<font style="color:rgba(0, 0, 0, 0.9);">在16位矩阵乘法过程中有效地使用量化权重</font>**<font style="color:rgba(0, 0, 0, 0.9);">，主要有以下三个创新点：</font>

    1. **<font style="color:rgba(0, 0, 0, 0.9);">4-bit NormalFloat（NF4）：</font>**<font style="color:rgb(25, 27, 31);">一</font><font style="color:#601BDE;">种新的数据类型4位NormalFloat（NF4</font><font style="color:rgb(25, 27, 31);">），</font><font style="color:#601BDE;">适合于正态分布数据</font><font style="color:rgba(0, 0, 0, 0.9);">的最佳量化数据类型。</font>
    2. **<font style="color:rgba(0, 0, 0, 0.9);">双重量化(Double Quantization—DQ)</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font><font style="color:#601BDE;">双重量化以减少平均内存占用</font><font style="color:rgb(25, 27, 31);">，</font><font style="color:rgba(0, 0, 0, 0.9);">使得在有限的硬件上微调更大的模型成为可能。</font>
    3. **<font style="color:rgba(0, 0, 0, 0.9);">分页优化器(Paged Optimizers)：</font>**<font style="color:#601BDE;">分页优化器来管理内存峰值。</font><font style="color:rgba(0, 0, 0, 0.9);"></font>

> <font style="color:rgb(5, 5, 5);">QLORA 有一种</font>**<font style="color:rgba(252,7,20,1);">低精度存储数据类型（4 bit），还有一种计算数据类型（BFloat16）</font>**<font style="color:rgb(5, 5, 5);">。实际上，这意味着无论何时使用 QLoRA 权重张量，我们都会将张量反量化为 BFloat16，然后执行 16 位矩阵乘法。</font>
>

### 技术方案
![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707114645902-ba60583f-f001-4da1-b334-ea22af73aa44.png)

#### <font style="color:rgb(0, 0, 0);">分块量化(Block-wise Quantization)</font>
**<font style="color:#601BDE;">全局量化</font>**<font style="color:#601BDE;">方式存在一个问题，当输入中存在极大值或者离群值时，一些较小的参数无法被精确的表示</font><font style="color:rgb(25, 27, 31);">，因此量化后的</font><font style="color:rgba(252,7,20,1);">神经网络效果会下降很多</font><font style="color:rgb(25, 27, 31);">。为了缓解这个问题，作者采用了</font><font style="color:#601BDE;">分块量化，即将输入划分为多个block，每个block分别量化</font><font style="color:rgb(25, 27, 31);">。全局量化和分块量化示意如下图所示：</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707103126979-9a5de96e-247e-452a-a7de-dc5d916e9111.png)

<font style="color:rgb(136, 136, 136);">分块量化</font>

<font style="color:rgb(25, 27, 31);">从图中可以看到，</font><font style="color:#601BDE;">分块量化能够明显减少量化过程中的误差(0.64 -> 0.241)</font><font style="color:rgb(25, 27, 31);">。</font>

#### <font style="color:rgb(25, 27, 31);">分位量化</font>
**<font style="color:rgb(25, 27, 31);">传统</font>****<font style="color:#601BDE;">线性量化</font>****<font style="color:rgb(25, 27, 31);">方法（对称量化）</font>**<font style="color:rgb(25, 27, 31);">将</font><font style="color:rgba(252,7,20,1);">原本不同的权重经量化后全数转化为相同的数值，导致模型出现较大误差</font>。<font style="color:#2F4BDA;">一般的模型参数通常</font>**<font style="color:#2F4BDA;">呈正态分布</font>**<font style="color:#2F4BDA;">，</font>**<font style="color:#2F4BDA;">而非均匀分布</font>**<font style="color:#2F4BDA;">。若依照线性方式进行量化，极可能导致多个不同的值被量化到相同的数值上</font>**<font style="color:#601BDE;">。</font>**

> **把量化理解成装格子的过程，int8量化有256个不同的选值，相当于有256个不同的格子（即，最大值和最小值中间，等间隔地摆放256个不同的格子），然后每个参数值都跳进离它最近的格子里。**如果原始数值相当接近，它们就极有可能最终跳入同一个"格子"**。如此一来，这些数值在量化后就会归并为一个，从而引起误差。**
>

![](https://cdn.nlark.com/yuque/0/2024/webp/35381469/1707104825749-9e50f075-0083-4f38-9608-11fe4491bf6f.webp)

**<font style="color:#601BDE;">分位量化(Quantile Quantization)</font>**<font style="color:rgb(25, 27, 31);">：</font>**<font style="color:rgb(25, 27, 31);">非对称量化方法</font>**<font style="color:rgb(25, 27, 31);">，以量化到4-bit为例，一共有16个数字可以选，那么可以先</font><font style="color:#601BDE;">将输入参数从小到大进行排序再等分为16份</font><font style="color:rgb(25, 27, 31);">，</font><font style="color:#601BDE;">每一份映射一个值</font><font style="color:rgb(25, 27, 31);">。这种分位量化方法量化出的参数就能</font><font style="color:#601BDE;">保证分布尽可能与原始分布相差不大。</font>

![](https://cdn.nlark.com/yuque/0/2024/webp/35381469/1707104825700-59ae6cc0-8e0c-4868-aced-352283c6dc03.webp)

**<font style="color:rgba(252,7,20,1);">问题：</font>****<font style="color:#601BDE;">分位量化会额外引入明显的计算开销</font>**<font style="color:rgb(25, 27, 31);">（每次有参数输入进来都需要对齐进行排序并等分）。提出NF方法。</font>

#### <font style="color:rgb(25, 27, 31);">NormFloat（NF）</font>
**<font style="color:rgba(252,7,20,1);">优化：基于参数服从正态分布假设，进行位置映射。具体地：</font>**<font style="color:#601BDE;">预训练的参数基本上都服从均值为0的正态分布，可以将其缩放到[-1, 1]的范围中</font><font style="color:rgb(25, 27, 31);">，将正态分布</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707115988333-fdd46f75-2b67-4781-9d90-a352cd53bc8d.png)<font style="color:rgb(25, 27, 31);">划分为</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707116064638-e2a774f8-8898-48fb-ab26-e58c5f14f120.png)<font style="color:rgb(25, 27, 31);">份，并缩放到[-1, 1]的范围中，</font><font style="color:#601BDE;">直接将参数映射到对应的分位，不用每次都对参数进行排序</font><font style="color:rgb(25, 27, 31);">。 此外，（</font>**<font style="color:rgb(25, 27, 31);">保证对称性</font>**<font style="color:rgb(25, 27, 31);">）作者分别</font>**<font style="color:#601BDE;">将负数和整数部分划</font>**<font style="color:rgb(25, 27, 31);">分为</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707116095634-5f305b7c-d7d1-4428-a3ad-3ea43aabc00a.png)<font style="color:rgb(25, 27, 31);">份，参数0还是放在0原本的位置上，解决参数0量化后可能不在0的位置上的问题。</font>

#### <font style="color:rgb(0, 0, 0);">NF4量化过程模拟</font>
![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707116662492-e5b2f36c-10b3-4cb5-83c4-0cf448868288.png)

    1. <font style="color:black;">输入分块，切分到不同block中</font>
    2. <font style="color:black;">找到输入的最大值，进行归一化</font>
    3. <font style="color:black;">找到最近的NF4分位值</font>

#### <font style="color:rgb(25, 27, 31);"> 双重量化</font>
![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1705995772267-7f8be1b0-801a-4de7-bab3-11e4da8cf1dd.png)

**<font style="color:rgba(252,7,20,1);">分块量化：</font>**<font style="color:#601BDE;">每个block都会额外产生一个量化常数</font><font style="color:#601BDE;">c</font><font style="color:rgb(25, 27, 31);">。以量化32bit参数、block大小64为例，每个block会引入32bit的量化常数，对应每个参数会额外引入32/64=0.5bit的额外开销。</font>

**<font style="color:rgba(252,7,20,1);">优化策略为：</font>**<font style="color:rgb(25, 27, 31);">在第一次量化后，并</font><font style="color:#601BDE;">不会直接储存量化常数</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707116465602-c8b5b19c-2900-4334-b313-11358e6f50b9.png)<font style="color:rgb(25, 27, 31);">，而是按照</font>**<font style="color:#74B602;">block大小256</font>**<font style="color:#601BDE;">对</font>**<font style="color:#601BDE;">量化常数</font>**<font style="color:#601BDE;">再</font>**<font style="color:#601BDE;">量化到8bit（</font>****<font style="color:#74B602;">FP8</font>****<font style="color:#601BDE;">）上去储存。</font>**

**<font style="color:#601BDE;">第一次量化：</font>**

<font style="color:rgb(25, 27, 31);">假设有</font>**<font style="color:#601BDE;"> 64 * 256 </font>**<font style="color:rgb(25, 27, 31);">个参数，blocksize 等于</font>**<font style="color:#601BDE;"> 64 </font>**<font style="color:rgb(25, 27, 31);">的情况下，会花费</font>**<font style="color:#601BDE;">256 个 32 bit</font>**<font style="color:rgb(25, 27, 31);"> 量化常量</font>**<font style="color:rgb(25, 27, 31);">，消耗</font>**![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1708413151387-25fb4ce8-38ff-4a86-9c9b-b881fbda8ae4.png)**<font style="color:rgb(25, 27, 31);">的空间</font>**<font style="color:rgb(25, 27, 31);">。那这 256 个 32bit 数据也挺花费空间的。</font>

**<font style="color:#601BDE;">第二次量化：</font>**

<font style="color:rgb(25, 27, 31);">对这 256 个数据进行进一步的量化，使用 </font>**<font style="color:#601BDE;">blocksize 为 256</font>**<font style="color:rgb(25, 27, 31);">，在花费掉一个32bit 量化常量</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1708413102099-d4fc6f61-62cc-4d6c-b3b6-0bbee647c78c.png)<font style="color:rgb(25, 27, 31);">，这 256 个 量化为 256 个 </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1708413115591-03adb8e0-7bca-48a0-9ce1-323c800003ad.png)<font style="color:rgb(25, 27, 31);">了，消耗掉 </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1708413167119-2c8accc3-1ad5-44a8-9a1c-39553b210b1b.png)<font style="color:rgb(25, 27, 31);"> 的空间。</font>

<font style="color:rgb(25, 27, 31);">将 </font><font style="color:rgb(25, 27, 31);">64∗256</font><font style="color:rgb(25, 27, 31);"> 个参</font>_<font style="color:rgb(25, 27, 31);">数</font>_<font style="color:rgb(25, 27, 31);">，进行一层量化的话，过程中消耗的空间为 </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1708413151387-25fb4ce8-38ff-4a86-9c9b-b881fbda8ae4.png)<font style="color:rgb(25, 27, 31);"> 的空间，进行二次量化的话，过程中会消耗掉 </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1708413167119-2c8accc3-1ad5-44a8-9a1c-39553b210b1b.png)<font style="color:rgb(25, 27, 31);"> 的空间。</font>

**<font style="color:#601BDE;">空间节省：</font>**

<font style="color:rgb(25, 27, 31);">对于每一个参数节约了</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1708413191702-446ab4f0-90c8-40bd-98b2-239158ca1e0e.png)

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707116412715-d8ce2d1a-a9ac-4058-b9fd-c9ec06bf9ef0.png?x-oss-process=image%2Fresize%2Cw_547%2Climit_0)



#### <font style="color:rgb(25, 27, 31);">分页优化</font>
**<font style="color:#601BDE;">分页优化</font>**<font style="color:rgb(25, 27, 31);">：针对</font><font style="color:#601BDE;">梯度检查点</font><font style="color:rgb(25, 27, 31);">做的进一步优化，以</font><font style="color:#601BDE;">防止在显存使用峰值时发生显存OOM的问</font><font style="color:rgb(25, 27, 31);">题。QLoRA分页优化其实就是</font><font style="color:#601BDE;">当显存不足是，将保存的部分梯度检查点转移到CPU内存上</font><font style="color:rgb(25, 27, 31);">，和计算机的内存数据转移到硬盘上的常规内存分页一个道理。</font><font style="color:#601BDE;">分页优化</font><font style="color:rgb(25, 27, 31);">在GPU显存不足的时候可以把optimizer转移到内存中，在需要更新optimizer状态时再加载回来，以此来减少GPU显存的峰值占用。</font>

#### QLORA与LORA对比
<font style="color:rgb(25, 27, 31);">QLoRA将微调65B参数模型的平均内存需求从>780GB的GPU内存降低到<48GB，而其运行时间和预测性能与16位完全微调基准相比并无损失。这标志着LLM微调的方式发生了重大转变。</font>

**<font style="color:rgb(25, 27, 31);">（一）LORA</font>**

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707191831959-a82b1938-f587-4c9f-ad51-ee0d58905f10.png)

**（二）QLoRA**

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707191863864-3493cd4d-0c0c-4db6-b94f-57dc7061658e.png)

**<font style="color:#601BDE;">反量化过程：</font>**

<font style="color:rgb(25, 27, 31);">以储存8-bit为例，QLoRA提供了去量化的公式</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1708310077231-ab803990-b722-4c79-80f8-33425c801f41.png)

> <font style="color:rgb(25, 27, 31);">去量化Step2， 使用</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1708310111013-65b2b8f6-3c90-462b-b3c2-34dd2f10c47c.png)<font style="color:rgb(25, 27, 31);">对256个</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1708310128604-8e2f253d-8b0e-43ae-a021-6da983ac819a.png)<font style="color:rgb(25, 27, 31);"> 恢复成256个FP32, 计为C_step2</font>
>
> <font style="color:rgb(25, 27, 31);">去量化Step1，使用256 个C_step2 去分别恢复对应block里64个4bit数字->64个FP32个数字，即为C_step1</font>
>
> <font style="color:rgb(25, 27, 31);">这样原始参数减去C_step1得到量化误差。</font>
>

### 代码实践参考
+ **<font style="color:rgba(0, 0, 0, 0.9);">bitsandbytes</font>**<font style="color:rgba(0, 0, 0, 0.9);">：该库包含量化大型语言模型（LLM）所需的所有工具。</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">Hugging Face Transformers 和 Accelerate</font>**<font style="color:rgba(0, 0, 0, 0.9);">：这些标准库用于 Hugging Face Hub 的高效模型训练。</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">PEFT</font>**<font style="color:rgba(0, 0, 0, 0.9);">：该库提供了各种方法的实现，以微调少量额外的模型参数。LoRA 需要它。</font>

#### NF4量化：
<font style="color:rgba(0, 0, 0, 0.9);">量化参数由</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>**<font style="color:rgba(0, 0, 0, 0.9);">BitsandbytesConfig</font>**<font style="color:rgba(0, 0, 0, 0.9);"> </font><font style="color:rgba(0, 0, 0, 0.9);">控制，如下所示：</font>

    - <font style="color:rgba(0, 0, 0, 0.9);">通过 </font>**<font style="color:rgba(0, 0, 0, 0.9);">load_in_4bit</font>**<font style="color:rgba(0, 0, 0, 0.9);"> 启用 4 位加载。</font>
    - **<font style="color:rgba(0, 0, 0, 0.9);">bnb_4bit_compute_dtype</font>**<font style="color:rgba(0, 0, 0, 0.9);"> 用于线性层计算的数据类型。</font>
    - <font style="color:rgba(0, 0, 0, 0.9);">嵌套量化通过 </font>**<font style="color:rgba(0, 0, 0, 0.9);">bnb_4bit_use_double_quant</font>**<font style="color:rgba(0, 0, 0, 0.9);"> 启用。</font>
    - **<font style="color:rgba(0, 0, 0, 0.9);">bnb_4bit_quant_type</font>**<font style="color:rgba(0, 0, 0, 0.9);"> 指定用于量化的数据类型。支持两种量化数据类型：</font>**<font style="color:rgba(0, 0, 0, 0.9);">fp4</font>**<font style="color:rgba(0, 0, 0, 0.9);"> （四位浮点）和 </font>**<font style="color:rgba(0, 0, 0, 0.9);">nf4</font>**<font style="color:rgba(0, 0, 0, 0.9);"> （常规四位浮点）。我们提倡使用 </font>**<font style="color:rgba(0, 0, 0, 0.9);">nf4</font>**<font style="color:rgba(0, 0, 0, 0.9);"> ，因为理论上它对于正态分布权重来说是最佳的。</font>

```python
model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path='/name/or/path/to/your/model',
        load_in_4bit=True,
        device_map='auto',
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        ),
    )
```

#### <font style="color:rgba(0, 0, 0, 0.9);"> QLoRA 高效微调</font>
```python
########## 环境
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git

########## 数据
from datasets import load_dataset

#dataset_name = "timdettmers/openassistant-guanaco" ###Human ,.,,,,,, ###Assistant

dataset_name = 'AlexanderDoria/novel17_test' #french novels
dataset = load_dataset(dataset_name, split="train")

########## 加载模型 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

model_name = "TinyPixel/Llama-2-7B-bf16-sharded"

### 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

###  模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

# 加载预训练模型的分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 创建一个 PEFT 配置对象，以便在训练和评估模型时使用。
from peft import LoraConfig, get_peft_model

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)

# 加载训练器
from transformers import TrainingArguments

output_dir = "./results"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 100
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 100
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)

# 然后最后将所有内容传递给训练器，创建一个训练器对象，以便对指定的语言模型进行训练。

# https://github.com/huggingface/trl
from trl import SFTTrainer

max_seq_length = 512
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)
trainer.train()
model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained("outputs")
```

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707193014973-460e1bb9-db9d-4e92-af73-772cd6ed4908.png)

```python
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig.from_pretrained('outputs')
model = get_peft_model(model, lora_config)

dataset['text']
['''### Human: Écrire un texte dans un style baroque,
utilisant le langage et la syntaxe du 17ème siècle, 
mettant en scène un échange entre un prêtre et un jeune homme 
confus au sujet de ses péchés.### Assistant: Si j'en luis éton. 
né ou empêché ce n'eſt pas ſans cauſe vů que ſouvent les liommes ne 
ſçaventque dire non plus que celui de tantôt qui ne ſçavoit rien faire 
que des civiéresVALDEN: Jefusbien einpêché confeſſant un jour un jeune 
Breton Vallonqui enfin de confeſſion me dit qu'il avoit beſongné une civiere .
Quoilui dis je mon amice peché n'eſt point écrit au livre Angeli que 
d'enfernommé la ſommedes pechez ,qui eſt le livre le plus déteſtable qui fut 
jamais fait& le plus blafphematoire d'autant qu'il eſt dédié à la plus femme
de bien je ne ſçai quelle penitence te donner ; mais non mon amiquel goûty prenois-tu
? Mon fieur bon & delectable. Quoi''']

text = "Écrire un texte dans un style baroque sur la glace et le feu ### Assistant: Si j'en luis éton"
device = "cuda:0"

inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# /usr/local/lib/python3.10/dist-packages/torch/utils/checkpoint.py:31: UserWarning: None of the inputs have requires_grad=True. Gradients will be None
  # warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")
Écrire un texte dans un style baroque sur la glace et le feu 
### Assistant: Si j'en luis étonné, si j'en brûle, si j'en frissonne, si j'en tremble, si j'en frémis, si j'en frissonne, si j'en frissonne, si
```

## **文档内容概述：QLoRA详解及代码实战**


本文档详细介绍了QLoRA（Quantized Low-Rank Adaptation）技术，这是一种高效的大型语言模型（LLM）微调方法。QLoRA的核心在于将预训练模型的参数量化到4位NormalFloat（NF4）精度进行存储，并在特征计算时通过双重量化（Double Quantization）将其还原到BFloat16精度进行参数更新。这种方法结合了低精度存储和高精度计算，使得在资源受限的硬件上微调更大的模型成为可能。





