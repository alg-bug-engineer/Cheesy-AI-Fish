# 11. Self-RAG 全代码详解

不看内容要代码版：[官方Repo](https://github.com/AkariAsai/self-rag/tree/main)

如何让大语言模型更加真实、正确、可靠？

检索增强生成（RAG）是一种有效的方法，可以减轻大型语言模型的局限性，例如幻觉和缺乏最新知识。

然而，如果你已经尝试过 RAG，可能会有相同的感受，那就是RAG上手容易，但很难达到可用的水平。

在这篇文章中，我将讨论一篇关于Self-RAG的新研究[论文](https://arxiv.org/abs/2310.11511)，该论文提出了一种提高 RAG 模型性能的新方法：

- 🚀 Self-RAG 7B 在 6/8 任务上优于 ChatGPT 3.5
- 🚀 在事实检查任务中准确率达到 81%

# Self-RAG是什么？

RAG 效率在很大程度上取决于文本分块方法、嵌入和检索技术等因素。

当不加选择地检索固定数量的段落或获取不相关的内容时，可能会导致不正确的结果，得到RAG在当场前场不可用的结论。

2023 年 10 月，一篇题为《Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection》的新研究论文介绍了一个创新框架。该框架通过指导LLM何时检索来提高LLM的一致性和表现。

实验表明，Self-RAG（7B 和 13B 参数）在多种任务上显着优于最先进的 LLM（例如 ChatGPT 和 Llama2-chat）和检索增强模型。

# Self-RAG：教LLM何时检索

自我反思检索增强生成（Self-RAG）是一种新方法，教导LLM检索、生成和批判，以提高其回答的真实性和质量。

与传统的检索增强生成 (RAG) 方法相比，Self-RAG 按需检索信息，这意味着它可以根据遇到的查询多次检索或根本不检索。

它还使用称为“反射令牌”的特殊令牌从各个角度评估其响应。这些令牌允许 LLM 在推理阶段控制其行为并根据特定任务要求进行定制。

![https://cdn.nlark.com/yuque/0/2023/png/406504/1699872645459-5038e607-22f5-4918-8c89-f459127beffb.png](https://cdn.nlark.com/yuque/0/2023/png/406504/1699872645459-5038e607-22f5-4918-8c89-f459127beffb.png)

Self-RAG根据不同的查询按需检索（例如，可以检索多次或完全跳过检索），并评价其自己的生成。

# Self-RAG 的内部运作

Self-RAG 方法的核心是通过“自我反思标记”来教导和控制大型语言模型 (LLM) 。这个过程发生在推理阶段，即LLM生成响应的时候。

以下是 Self-RAG 的工作原理：

1. 检索：Self-RAG 首先确定将检索到的信息添加到响应中是否会有帮助。如果是，则它发出检索过程的信号，并要求外部检索模块查找相关文档。
2. 生成：如果不需要检索，Self-RAG 会预测响应的下一部分，就像常规语言模型一样。如果需要检索，它首先评估检索到的文档是否相关，然后根据发现的内容生成响应的下一部分。
3. 评价：如果需要检索，Self-RAG 会检查检索到的段落是否支持响应。它还评估响应的整体质量。

# Self-RAG的训练过程

Self-RAG 训练两个模型，评论家和生成者，两者都使用反射标记扩展标记词汇表，并使用标准的下一个标记预测目标进行训练。

- 第 1 步 Critic 数据创建：通过提示 GPT-4 生成反射标记来生成 Critic 训练数据。
- 步骤 2 Critic 训练：在合成数据集上训练 Critic 模型。
- 步骤 3 生成器数据创建：使用 Critic 和 Retriever 生成生成器训练数据。
- 第 4 步生成器训练：在 RAG 数据集上训练生成器，包括用于教导模型何时检索或不检索的特殊标记。

# 应用Self-RAG

# 使用 Self-RAG 运行推理

Self-RAG 模型在基础模型 Llama-2–7b-hf 上进行了微调，现已在 Hugging Face 上提供。

以下是如何使用 Self-RAG 运行推理。

1. 安装必要的包

```
!pip  install  vllm
!pip  install  torch
!pip  install  Transformers,  tokenizer,  datasets,  peft,  Bitsandbytes
!pip  install  Accelerator>=0.21.0,<0.23.0
!pip  install  peft>=0.4.0
!pip  installvaluate  >=0.4.0
!pip  install  tiktoken
```

2. 运行推理

按照研究团队的建议使用 vllm 来加速推理。

```
# Import necessary libraries
from vllm import LLM, SamplingParams

# Initialize the LLM (Large Language Model) with the specified model and data type
model = LLM("selfrag/selfrag_llama2_7b", dtype="half")

# Define sampling parameters for text generation
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)

# Define a function to format prompts, including an instruction and an optional paragraph for retrieval
def format_prompt(input, paragraph=None):
  prompt = "### Instruction:\n{0}\n\n### Response:\n".format(input)
  if paragraph is not None:
    prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
  return prompt

# Define two queries for the model to generate responses for
query_1 = "Leave odd one out: twitter, instagram, whatsapp."
query_2 = "Can you tell me the difference between llamas and alpacas?"
queries = [query_1, query_2]

# Generate responses for the queries
preds = model.generate([format_prompt(query) for query in queries], sampling_params)

# Print the model's predictions for each query
for pred in preds:
  print("\n\nModel prediction: {0}".format(pred.outputs[0].text))
```

我们来分析一下 Self-RAG 的输出。

我们执行了两个查询，并观察到对于第一个查询，Self-RAG 直接生成响应，而不检索信息，因为检索是不必要的。在第二个查询中，Self-RAG 显示 [Retrieve] 标记，因为问题需要更多事实信息。

![https://cdn.nlark.com/yuque/0/2023/png/406504/1699872645265-60f32403-1e6e-40e9-a163-cdee8a96f181.png](https://cdn.nlark.com/yuque/0/2023/png/406504/1699872645265-60f32403-1e6e-40e9-a163-cdee8a96f181.png)

第一个查询没有检索，第二个查询有信息检索。图片由作者提供。

# 使用自己的数据运行 Self-RAG

在这里，我们将使用自己的数据与 Self-RAG 交互的两种方法：

1. 直接数据插入：最简单的方法是将数据直接插入到函数paragraph的参数中format_prompt。这使得 Self-RAG 可以直接访问你的数据并将其合并到其响应生成过程中。
2. 基于嵌入的检索：对于更复杂的数据或直接插入不切实际的情况，可以为数据生成嵌入，并利用检索机制为 Self-RAG 提取相关信息。这种方法使 Self-RAG 能够根据查询上下文有选择地检索和利用数据中的信息。

# 直接数据插入

对于需要外部事实依据的查询，可以插入一个段落。

Self-RAG 可以在生成时随时检索和插入段落，并且只要它们被上下文标记特殊标记包围即可识别它们<paragraph>。</paragraph>

![https://cdn.nlark.com/yuque/0/2023/png/406504/1699872645084-85890db7-650c-43a1-aead-87fe12cf07d4.png](https://cdn.nlark.com/yuque/0/2023/png/406504/1699872645084-85890db7-650c-43a1-aead-87fe12cf07d4.png)

通过“paragraph”参数添加事实。图片由作者提供。

# 基于嵌入的检索

可以利用原始研究论文提供的[脚本和代码](https://github.com/AkariAsai/self-rag/tree/main/retrieval_lm)将自己的数据合并到 Self-RAG 中。该过程涉及三个主要步骤：

1. 数据准备

以 JSON 或 JSONL 格式准备数据集。每个实例应包含一个问题或一条指令，这将作为检索期间的查询。

2. 嵌入生成

使用 GitHub 存储库中提供的脚本为您的数据生成嵌入。这会将你的数据转换为可由检索机制有效处理的数字表示形式。

```
cd retrieval_lm
CUDA_VISIBLE_DEVICES=0

python generate_passage_embeddings.py  --model_name_or_path facebook/contriever-msmarco \
--output_dir [YOUR_OUTPUT_DIR] \
--passages [YOUR_PASSAGE_DATA] \
--shard_id 0  --num_shards 4 > ./log/nohup.my_embeddings 2>&1 &
```

3. 对自己的数据运行 Self-RAG

将生成的嵌入与 Self-RAG 的检索模块相结合，从数据中提取相关信息。然后，Self-RAG 将利用此信息生成针对你的特定查询的响应。

```
from passage_retriever import Retriever

retriever = Retriever({})
query = "YOUR_QUERY"
retriever.setup_retriever("facebook/contriever-msmarco", [YOUR_JSON_DATA_FILE], [YOUR_DATA_EMBEDDING],  n_docs=5, save_or_load_index=False)
retrieved_documents = retriever.search_document(query, 5)
prompts = [format_prompt(query, doc["title"] +"\n"+ doc["text"]) for doc in retrieved_documents]
preds = model.generate(prompts, sampling_params)
top_doc = retriever.search_document(query, 1)[0]
print("Reference: {0}\nModel prediction: {1}".format(top_doc["title"] + "\n" + top_doc["text"], preds[0].outputs[0].text))
```

# 使用 Self-RAG 增强 RAG Pipeline

或者，也可以使用 Self-RAG 方案训练新的预训练 LLM，例如 ChatGLM3-6b。官方仓库已经提供了比较好用的微调训练[脚本](https://github.com/AkariAsai/self-rag/blob/main/retrieval_lm/script_finetune_7b.sh)。

# 最后

RAG 可能会导致不恰当的响应。在此过程中添加特殊标记可以教导模型何时检索信息以及相关信息。

Self-RAG 是一种有趣的方法，可以在不牺牲大型语言模型太多创造性多功能性的情况下提高事实性。它显示了让LLM反思自身局限性并有选择地获取知识的好处。

论文中分享的模型已经在 150,000 个实例的评价者训练中进行了微调。这是一个具挑战性的任务，因为检索和非检索示例的数据集都需要多样化。最酷的是，完整的数据集和代码在 GitHub 上完全共享，因此可以微调任意的自定义 LLM，例如 ChatGLM3-6b、baichuan等。