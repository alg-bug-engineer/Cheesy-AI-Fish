![https://cdn.nlark.com/yuque/0/2023/png/406504/1703729850109-8070a7ba-c613-487f-9b31-dc435b3a68e5.png](https://cdn.nlark.com/yuque/0/2023/png/406504/1703729850109-8070a7ba-c613-487f-9b31-dc435b3a68e5.png)

在预训练模型的每一层或某些层中添加Adapter模块，Adapter模块由两个前馈神经网络组成，第一个FNN将Transformer的输出D维映射到m维度，一般m<<D，第二个FNN将m映射到D输出，通过控制m的大小来限制Adapter的参数量；通过添加Adapter模块来产生一个易于扩展的下游模型，每当出现新的下游任务，通过添加Adapter模块来避免全模型微调与灾难性遗忘的问题。Adapter方法不需要微调预训练模型的全部参数，通过引入少量针对特定任务的参数，来存储有关该任务的知识，降低对模型微调的算力要求。

![https://cdn.nlark.com/yuque/0/2023/png/406504/1703730495994-ae9f1afa-dda4-4e42-90c7-94c8d7631c84.png](https://cdn.nlark.com/yuque/0/2023/png/406504/1703730495994-ae9f1afa-dda4-4e42-90c7-94c8d7631c84.png)

![https://cdn.nlark.com/yuque/0/2023/png/406504/1703730502897-843e860d-a239-4bd9-8a20-91fc04d069f3.png](https://cdn.nlark.com/yuque/0/2023/png/406504/1703730502897-843e860d-a239-4bd9-8a20-91fc04d069f3.png)

特定任务上，通过微调少量参数，提升模型效果，是一种侵入式的方法，需要修改原始模型实现