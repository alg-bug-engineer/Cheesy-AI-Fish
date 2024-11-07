## LORA
### 背景
<font style="color:#8A8F8D;">论文标题:LoRA: Low-Rank Adaptation of Large Language Models</font>

<font style="color:#8A8F8D;">论文链接:https://arxiv.org/pdf/2106.09685.pdf</font>

[Aghajanyan](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2012.13255)<font style="color:rgb(25, 27, 31);">的研究表明：</font>**<font style="color:rgb(25, 27, 31);">预训练模型拥有极小的内在维度(instrisic dimension)，即存在一个极低维度的参数，微调它和在全参数空间中微调能起到相同的效果</font>**<font style="color:rgb(25, 27, 31);">。同时发现在预训练后，</font><font style="color:#601BDE;">越大的模型有越小的内在维度</font><font style="color:rgb(25, 27, 31);">，这也解释了为何大模型都拥有很好的few-shot能力。</font>

### <font style="color:rgb(25, 27, 31);">出发点</font>
<font style="color:rgb(0, 0, 0);">随着大模型的发展，尤其是chatGPT出现之后，175B的参数量，训练起来非常昂贵。</font>

<font style="color:#DF2A3F;">微软</font><font style="color:rgb(0, 0, 0);">提出了低秩自适应（LoRA），</font>**<font style="color:rgb(0, 0, 0);">LoRA的主要思想很简单，冻结预训练模型的权重参数，</font>**<font style="color:rgb(18, 18, 18);">在</font><font style="color:#601BDE;">原始的预训练模型（PLM）旁边增加一个新的通路，通过前后两个矩阵A,B相乘</font>**<font style="color:rgb(0, 0, 0);">，</font>**<font style="color:#601BDE;">第一个矩阵A负责降维，第二个矩阵B负责升维，中间层维度为r，</font>**<font style="color:rgb(0, 0, 0);">在微调下游任务的时候，只更新A和B</font>**<font style="color:rgb(0, 0, 0);">，该方法的核心思想就是</font>**<font style="color:#601BDE;">通过低秩分解来模拟参数的改变量</font>**<font style="color:rgb(0, 0, 0);">，从而以</font><font style="color:#601BDE;">极小的参数量</font><font style="color:rgb(0, 0, 0);">来实现大模型的间接训练。</font>

<font style="color:rgb(0, 0, 0);">LoRA方法优点：</font>

    - **<font style="color:rgb(1, 1, 1);">预训练模型可以共享</font>**<font style="color:rgb(1, 1, 1);">，针对</font><font style="color:#601BDE;">下游任务可以</font>**<font style="color:#601BDE;">构建多个不同任务的LoRA模块</font>**<font style="color:rgb(1, 1, 1);">。冻结预训练模型参数共享，通过替换矩阵A和B来高效地切换任务，</font><font style="color:#601BDE;">从而</font>**<font style="color:#601BDE;">显著降低存储需求和任务切换开销</font>****<font style="color:rgb(1, 1, 1);">。</font>**
    - <font style="color:rgb(1, 1, 1);">LoRA使训练更加的高效，将</font>**<font style="color:#601BDE;">硬件的进入门槛降低了3倍</font>**<font style="color:rgb(1, 1, 1);">，相同的内存下，可以微调更大参数的模型</font>
    - <font style="color:#601BDE;">线性设计允许我们在部署时将可训练矩阵与冻结权重合并</font><font style="color:rgb(1, 1, 1);">，和完全微调的模型相比，</font>**<font style="color:#601BDE;">不会引入推理延迟。</font>**

### <font style="color:rgb(0, 0, 0);">具体实现方法</font>
![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707193479465-c9540e0f-effd-425e-8458-723003a9ab79.png)

#### 公式
**<font style="color:#DF2A3F;">LoRA中是让模型学习BA，去近似SVD分解的结果</font>**

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707097562373-da7340db-56a1-4a58-8de9-2f4b9c99345b.png)

+ <font style="color:rgb(0, 0, 0);">在训练过程中</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707097571222-015cda8e-6347-4545-b8bc-ebc456748d52.png)<font style="color:rgb(0, 0, 0);">被冻结，</font><font style="color:#601BDE;">不接收梯度更新</font><font style="color:rgb(0, 0, 0);">，而</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707097586640-2c079db4-67d7-4a49-9bc7-5fcf8dfee746.png)<font style="color:rgb(0, 0, 0);">和</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707097598438-0a0b99ee-e750-411c-ad2c-721300f8145a.png)<font style="color:rgb(0, 0, 0);">包含可训练参数。</font>
+ <font style="color:rgb(0, 0, 0);">在初始化的时候，我们对使用</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707097586640-2c079db4-67d7-4a49-9bc7-5fcf8dfee746.png)<font style="color:#601BDE;">随机高斯初始化</font><font style="color:rgb(0, 0, 0);">，对</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707097598438-0a0b99ee-e750-411c-ad2c-721300f8145a.png)<font style="color:rgb(0, 0, 0);">使用</font><font style="color:#601BDE;">零初始化</font><font style="color:rgb(0, 0, 0);">。</font>
+ <font style="color:rgb(0, 0, 0);">因此，在训练开始时为0，然后我们可以通过</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707097637943-39593fbc-83e7-41a9-85a4-a1a7ba371c5e.png)<font style="color:rgb(0, 0, 0);">对</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707097655450-a6a4b554-a118-4d1d-a9a8-095fd2c87013.png)<font style="color:rgb(0, 0, 0);">进行缩放，</font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707097706524-a9045b07-4643-4dae-9ae2-bd69480be8f4.png)<font style="color:rgb(0, 0, 0);">是秩 在推理时，我们通过上图可知，将左右两部分进行相加即可，不会添加额外的计算资源。</font>

**<font style="color:rgb(25, 27, 31);">在微调过程中，所有做lora适配器的module，它们的</font>**![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707097706524-a9045b07-4643-4dae-9ae2-bd69480be8f4.png)**<font style="color:rgb(25, 27, 31);">都是一致的，且在训练过程中不会改变</font>**<font style="color:rgb(25, 27, 31);">。</font>

<font style="color:rgb(18, 18, 18);">Transformer的权重矩阵包括Attention模块里用于计算query, key, value的Wq，Wk，Wv以及多头attention的Wo,以及MLP层的权重矩阵，</font>**<font style="color:rgba(252,7,20,1);">在LoRA原始论文中，作者</font>**<font style="color:rgba(252,7,20,1);">通过消融实验发现</font>**<font style="color:rgb(25, 27, 31);">最</font>****<font style="color:#601BDE;">终选择对attention模块的</font>**<font style="color:#601BDE;"> </font>![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707098193512-e51497be-88fd-49c9-b3ef-0fc24b4695df.png)<font style="color:#601BDE;"> </font>**<font style="color:#601BDE;">做低秩适配</font>**<font style="color:rgb(18, 18, 18);">产生最佳结果</font><font style="color:#601BDE;">。</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707193655603-3f6b31f1-d495-40ad-aded-56fdaae34520.png)![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707193677629-3de1fc8b-314e-4654-8be7-278585d172c0.png)

![](https://cdn.nlark.com/yuque/0/2024/png/35381469/1707200971223-79f9009f-76ec-45e0-b5ca-7a8f5e08ed95.png)

### <font style="color:rgb(0, 0, 0);">4.代码实战</font>
```python
from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType

import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset


device = "cuda"

model_name_or_path = "/data/nfs/llm/model/bloomz-560m"
tokenizer_name_or_path = "/data/nfs/llm/model/bloomz-560m"

####  加载配置 ################
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                         inference_mode=False, r=8, 
                         lora_alpha=32, 
                         lora_dropout=0.1)

dataset_name = "twitter_complaints"
checkpoint_name = f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_v1.pt".replace("/", "_")
text_column = "Tweet text"
label_column = "text_label"
max_length = 64
lr = 3e-2
num_epochs = 10
batch_size = 8


from datasets import load_dataset

# dataset = load_dataset("ought/raft", dataset_name)
dataset = load_dataset("/home/guodong.li/data/peft/raft/raft.py", dataset_name, cache_dir="/home/guodong.li/data/peft/data")

classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
print(classes)
dataset = dataset.map(
    lambda x: {"text_label": [classes[label] for label in x["Label"]]},
    batched=True,
    num_proc=1,
)
print(dataset)
dataset["train"][0]


# data preprocessing
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes])
print("target_max_length:", target_max_length)


def preprocess_function(examples):
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["train"]


train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
def test_preprocess_function(examples):
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    model_inputs = tokenizer(inputs)
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (max_length - len(sample_input_ids)) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs["attention_mask"][i]
        
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
    return model_inputs


test_dataset = dataset["test"].map(
    test_preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

test_dataloader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
next(iter(test_dataloader))

#####  加载模型 ############
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# model
# optimizer and lr scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)


# training and evaluation
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        #         print(batch)
        #         print(batch["input_ids"].shape)
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

第四步，模型训练的其余部分均无需更改，
当模型训练完成之后，保存高效微调部分的模型权重以供模型推理即可。
peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
model.save_pretrained(peft_model_id)
```

```python

第五步，加载微调后的权重文件进行推理。
from peft import PeftModel, PeftConfig

peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
config = PeftConfig.from_pretrained(peft_model_id)
# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
# 加载PEFT模型
model = PeftModel.from_pretrained(model, peft_model_id)

# tokenizer编码
inputs = tokenizer(f'{text_column} : {dataset["test"][i]["Tweet text"]} Label : ', return_tensors="pt")

# 模型推理
outputs = model.generate(
        input_ids=inputs["input_ids"], 
        attention_mask=inputs["attention_mask"], 
        max_new_tokens=10, 
        eos_token_id=3
    )

# tokenizer解码
print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

```

