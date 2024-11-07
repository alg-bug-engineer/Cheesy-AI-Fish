LangChain 是一个由大语言模型驱动的应用开发框架。

LangChain 原来主要的模块有 6 个，分别是模型 Models、提示 Prompt、索引 Indexes、链 Chains、代理 Agents 和记忆 Memory。

后来经过调整，目前主要的模块有 6 个，分别是模型 Model I/O（原来的 Prompts 被归类到了 Model I/O 中）、检索 Retrieval（实际上就是原来的 Indexes 模块）、链 Chains、代理 Agents、记忆 Memory和新增加的回调 Callbacks。每个模块都有不同的作用，这里我简单介绍一下它们的作用。  

Models 模块主要是对 LLM 层进行抽象。可以简化我们调用 LLM 接口的过程，这样我们可以很轻易的替换底层的 LLM。Models 目前分为两类：
- 一类是普通 LLM，LLM 一般是输入文本字符串并返回文本字符串的模型，比如 text-davinci-003。
- 一类是 Chat Model，这类模型通常支持传入消息列表作为输入并返回的模型，比如 GPT-3.5、GPT-4 和 Claude 等。 

Prompts 模块主要是提供了一些工具，方便我们更容易构建需要的提示词。这部分提供的工具有 PromptTemplates、ChatPrompt Templates 和 Example Selectors。
- PromptTemplates 主要就是通过模板的方式让我们更好的复用提示词。我们给定一个模板字符串，再从用户输入中获取一组参数进行生成，得到最终的提示。主要包含语言模型的说明，也就是模型当前扮演的角色，一组少量的示例，可以帮助 LLM 完成零样本学习，更好的生成响应，最后就是具体的问题需求。
- ChatPrompt Templates 是针对聊天模型设计的，功能与 PromptTemplates 类似，不过可以接受消息列表作为输入，列表会包含一组消息，每个消息都会有一个角色。
- Example Selectors 是对示例的选择。我们可以定义很多示例，然后根据不同的规则去选择示例。比如根据输入长度选择示例，输入长时选择多一些示例，输入短时少选择一些示例。也可以根据相关性来选择和输入最相关的示例。  

Indexes 模块主要作用就是和外部数据进行集成，根据外部数据来获取答案。索引主要包含 Document Loaders、Text Splitters、Vectorstores 和 Retrievers 几个模块。使用它们可以完成从外部获取数据的标准步骤。
1. 首先通过 Document Loaders 模块加载不同类型的数据源，比如网络的网页、本地的文档等。
2. 第二步是通过 Text Splitters 把第一步获取的文本进行分割，因为 LLM 都会限制上下文窗口大小，有 4k、16k、32k 等。把文本分割成制定大小的 chunk，可以更好的控制 token 大小。
3. 第三步就是通过 Vectorstores 模块把文本转换为向量，并存入向量数据库中。我们可以使用 OpenAI 的 Embedding API，也可以使用 HuggingFace 的 Embeddings 加载本地模型，可以节省调用费用，提高数据安全性。
4. 最后一步通过 Retriever 模块根据相关性从向量数据库中检索对应的文档。 

Chains 模块通过 Chain 的概念，来将各个组件进行链接，来简化开发复杂应用的难度。Chain 有很多种类型，主要的有 LLM Chain、Sequential Chain 和 Router Chain。
- LLM Chain 是最常用的，由 PromptTemplate、LLM 和 OutputParser 组成。LLM 输出是文本，OutputParser 可以让 LLM 结构化输出并且进行结果解析，比如指定输出为 JSON 格式，这样方便后续的处理。
- Sequential Chain 是顺序链，可以完成多个步骤的任务，每个步骤都有一个或多个输入和输出，上一个步骤的输出就是下一个步骤的输入，有点像编程中管道的概念。比如我们可以先将文本进行翻译，再对翻译后的文本进行总结。
- Router Chain 更加复杂，它是根据输入来动态选择下一个 chain。它由两个组件组成，分别是路由器链本身，负责选择调用下一个链。
	- 路由器链又分两种，一种是 LLM Router Chain，由 LLM 做路由决策，
	- 一种是 Embeding Router Chain，通过向量搜索做路由决策。
	- 另一个组件是目标链列表，也就是路由器链可以路由到的子链。  

Agent 模块也就是代理模块。在代理出现之前，我们使用 LLM 的方式通常是给定一个较为复杂的目标，LLM 无法一次性完成，然后我们还需要设计为了达成目标的每一个步骤的 Prompt。
代理模式就是我们只需要给定一个目标，由 LLM 自己去思考每个步骤。代理模块主要由 4 个组件组成，分别是代理 Agent、工具 Tools、工具包 Toolkits 和代理执行器 Agent Executor。
- Agent 的作用是调用 LLM，获取下一步的 Action；
- Tools 是 Agent 可以调用的方法列表，LangChain 内部有很多工具，比如查询数据库、发送邮件、处理 JSON 等等，Tool 有一个 description 属性，LLM 会通过这个属性来决定是否使用这个工具；
- ToolKits 是一组工具集，为了实现某个特定目标而提供的工具集合；
- Agent Executor 负责迭代运行代理的每个 Action。
从功能上看，Agent 模块和 Chains 模块的功能有些相似，不过 Chains 是由开发者预先定义好一系列 Action，再由 LLM 从其中选择最合适的 Action。Agent 是由 LLM 自己来定义并执行 Action。所以我们可以把 Agent 理解成自由度和随机性更高的 Chain。  

Memory 模块也叫记忆模块或者存储模块，它的主要作用是用来存储之前交互的信息。无论是 Chain 还是 Agent，每次交互都是==无状态的==，我们无法在当前交互中得知之前历史交互的信息。
Memory 就是一种可以跨多轮交互提供历史上下文的能力。Memory 支持在多种存储中存储历史数据，比如 MongoDB、SQLite 和 Redis 等。它还支持通过 Buffer Memory 直接在内存中存储信息。存储的形式主要有三种，
1. 第一种是 ConversationSummaryMemory，也就是用摘要的形式保存记录；
2. 第二种是 ConversationBufferWindowMemory，用原始形式保存最近的 N 条记录；
3. 第三种是 ConversationBufferMemory，用原始形式保存所有记录。  

Callbacks 模块的主要作用就是在 LLM 执行的各个流程环节中插入回调函数，来获取整个 LLM 执行的所有参数，适合做监控、日志记录等工作。相比较其他模块，回调函数可以在任意的环节进行，比如产生新的 Token、链的开始和结束、代理 Action 的开始和结束、Tool 的开始和结束等。


--------
#LLM #LangChain
