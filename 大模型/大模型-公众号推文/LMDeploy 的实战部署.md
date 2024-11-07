LMDeploy 由 MMDeploy 和 MMRazor 团队联合开发，是涵盖了 LLM 任务的全套轻量化、部署和服务解决方案。 这个强大的工具箱提供以下核心功能：

+ 高效推理引擎 TurboMind：基于 FasterTransformer，我们实现了高效推理引擎 TurboMind，支持 InternLM、LLaMA、vicuna等模型在 NVIDIA GPU 上的推理。
+ 交互推理方式：通过缓存多轮对话过程中 attention 的 k/v，记住对话历史，从而避免重复处理历史会话。
+ 多 GPU 部署和量化：我们提供了全面的模型部署和量化支持，已在不同规模上完成验证。
+ persistent batch 推理：进一步优化模型执行效率。![](https://cdn.nlark.com/yuque/0/2023/gif/406504/1689077790639-0051c7d4-9fba-4f94-ac86-e4ef81fd4487.gif)

## 性能
如下图所示，我们对比了 facebookresearch/llama、HuggingFace Transformers、DeepSpeed 在 7B 模型上的token生成的速度。

测试设备：NVIDIA A100(80G)

测试指标：吞吐量（token/s)

测试数据：输入token数为1，生成token数为2048

TurboMind 的吞吐量超过 2000 token/s, 整体比 DeepSpeed 提升约 5% - 15%，比 huggingface transformers 提升 2.3 倍

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1689077791363-b2544613-3de9-40da-84ba-60fbed68bd6c.png)

## 快速上手
### 安装
conda create -n lmdeploy python=3.10 conda activate lmdeploy git clone https://github.com/InternLM/lmdeploy.git cd lmdeploy pip install -e .

### 部署 InternLM
#### 获取 InternLM 模型
# 1. 下载 InternLM 模型 # Make sure you have git-lfs installed (https://git-lfs.com) git lfs install git clone https://huggingface.co/internlm/internlm-7b /path/to/internlm-7b # if you want to clone without large files – just their pointers # prepend your git clone with the following env var: GIT_LFS_SKIP_SMUDGE=1 # 2. 转换为 trubomind 要求的格式。默认存放路径为 ./workspace python3 -m lmdeploy.serve.turbomind.deploy internlm-7b /path/to/internlm-7b hf 

#### 使用 turbomind 推理
docker run --gpus all --rm -v $(pwd)/workspace:/workspace -it openmmlab/lmdeploy:latest \     python3 -m lmdeploy.turbomind.chat internlm /workspace



```plain
turbomind 在使用 FP16 精度推理 InternLM-7B 模型时，显存开销至少需要 22.7G。建议使用 3090, V100，A100等型号的显卡
```



#### 部署推理服务
使用下面的命令启动推理服务：

bash workspace/service_docker_up.sh

你可以通过命令行方式与推理服务进行对话：

python3 lmdeploy.serve.client {server_ip_addresss}:33337 internlm

也可以通过 WebUI 方式来对话：



```plain
python3 lmdeploy.app {server_ip_addresss}:33337 internlm
```



![]()

其他模型的部署方式，比如 LLaMA，vicuna，请参考[这里](https://github.com/InternLM/lmdeploy/blob/main/docs/zh_cn/serving.md)

### 基于 PyTorch 的推理
#### 单个 GPU
python3 -m lmdeploy.pytorch.chat $NAME_OR_PATH_TO_HF_MODEL\     --max_new_tokens 64 \     --temperture 0.8 \     --top_p 0.95 \     --seed 0

#### 使用 DeepSpeed 实现张量并行
deepspeed --module --num_gpus 2 lmdeploy.pytorch.chat \     $NAME_OR_PATH_TO_HF_MODEL \     --max_new_tokens 64 \     --temperture 0.8 \     --top_p 0.95 \     --seed 0

## 量化部署
在 fp16 模式下，可以开启 kv_cache int8 量化，单卡可服务更多用户。 首先执行量化脚本，量化参数存放到 deploy.py 转换的 workspace/triton_models/weights 目录下。



```plain
python3 -m lmdeploy.lite.apis.kv_qparams \
  --model $HF_MODEL \
  --output_dir $DEPLOY_WEIGHT_DIR \
  --symmetry True \ # 对称量化或非对称量化，默认为 True
  --offload  False \ # 将模型放在 CPU，只在推理时加载部分模块到 GPU，默认为 False
  --num_tp 1  \  # Tensor 并行使用的 GPU 数，和 deploy.py 保持一致
```



然后调整 workspace/triton_models/weights/config.ini

+ use_context_fmha 改为 0，表示关闭
+ quant_policy 设置为 4。此参数默认为 0，表示不开启

