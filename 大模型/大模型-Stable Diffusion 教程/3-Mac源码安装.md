## <font style="color:rgb(52, 73, 94);">1.作品图</font>
![](https://cdn.nlark.com/yuque/0/2023/png/406504/1689125033641-06c5a3f3-ba2e-40ed-9de5-0840f45356d6.png)

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1689125033514-55ad9d14-2d08-4304-88b6-cae482a312d2.png)

## <font style="color:rgb(52, 73, 94);">2.准备工作</font>
<font style="color:rgb(52, 73, 94);">目前网上能搜到的stable-diffusion-webui的安装教程都是Window和Mac M1芯片的，而对于因特尔芯片的文章少之又少，这就导致我们还在用老Intel 芯片的Mac本，看着别人生成美女图片只能眼馋。所以小卷这周末折腾了一天，总算是让老Mac本发挥作用了。先来说说准备工作：</font>

+ <font style="color:rgb(52, 73, 94);">Mac笔记本操作系统版本 >= 13.2.1 （亲测10.0版本各种问题无法运行，无奈花了一小时升级系统）</font>
+ <font style="color:rgb(52, 73, 94);">Python3.10.6版本（已安装其他版本也不要紧，后面我们用Conda做版本控制）</font>
+ <font style="color:rgb(52, 73, 94);">stable-diffusion-webui代码下载，下载地址：</font>[https://github.com/AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## <font style="color:rgb(52, 73, 94);">3.安装步骤</font>
### <font style="color:rgb(52, 73, 94);">3.1 依赖安装</font>
<font style="color:rgb(52, 73, 94);">从github上把stable-diffusion-webui的源代码下载下来，进入到stable-diffusion-webui目录下，执行</font>

**<font style="color:rgb(52, 73, 94);">python</font>**



```python
pip install -r requirements_versions.txt
```



<font style="color:rgb(52, 73, 94);">这一步是安装Python项目运行所有需要的依赖，这步很大概率出现无法安装gfpgan的问题：Couldn’t install gfpgan</font>

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1689125031297-9ab17195-bad9-4a3c-80ac-8dee1513f611.png)

<font style="color:rgb(52, 73, 94);">解决方法：</font>

<font style="color:rgb(52, 73, 94);">网络连接超时的问题，更改pip使用阿里源</font>

### <font style="color:rgb(52, 73, 94);">3.2pip更换国内镜像库</font>
<font style="color:rgb(52, 73, 94);">更换方法参考：</font>[https://blog.csdn.net/qq_45770232/article/details/126472610](https://blog.csdn.net/qq_45770232/article/details/126472610)

### <font style="color:rgb(52, 73, 94);">3.3安装anaconda</font>
<font style="color:rgb(52, 73, 94);">这一步是方便对Python做版本控制，避免卸载重新安装不同版本的Python。</font>

<font style="color:rgb(52, 73, 94);">下载安装地址：</font>[https://www.anaconda.com/](https://www.anaconda.com/)

<font style="color:rgb(52, 73, 94);">从官网下载一路点击安装就行。</font>

#### <font style="color:rgb(52, 73, 94);">Conda添加环境变量</font>
<font style="color:rgb(52, 73, 94);">安装完成后，打开终端，输入conda，如果是无法识别的命令。需要配置环境变量，配置方法：</font>

<font style="color:rgb(52, 73, 94);">修改</font><font style="color:rgb(233, 105, 0);background-color:rgb(248, 248, 248);">.bash_profile</font><font style="color:rgb(52, 73, 94);">添加自己安装conda的路径，命令如下：</font>

```shell
vim ~/.bash_profile

# 打开文件后，写入下面这行到文件里，注意替换路径
export PATH="/Users/(你自己的路径)/anaconda3/bin:$PATH"
```

<font style="color:rgb(52, 73, 94);">接着</font><font style="color:rgb(233, 105, 0);background-color:rgb(248, 248, 248);">:wq</font><font style="color:rgb(52, 73, 94);">保存退出，</font><font style="color:rgb(233, 105, 0);background-color:rgb(248, 248, 248);">source ~/.bash_profile</font><font style="color:rgb(52, 73, 94);">使配置生效</font>

#### <font style="color:rgb(52, 73, 94);">修改conda源为国内镜像库</font>
<font style="color:rgb(52, 73, 94);">执行命令如下：</font>

```shell
# 如果没有会创建condarc文件
vim ~/.condarc

# 打开文件后，把下面的内容粘贴进去保存
channels:
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
  - https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - defaults
show_channel_urls: true
```

### <font style="color:rgb(52, 73, 94);">3.4 创建虚拟环境</font>
<font style="color:rgb(52, 73, 94);">执行命令：</font>

```shell
conda create -n sd python=3.10.6
```

<font style="color:rgb(52, 73, 94);">这样就创建了一个名称为</font><font style="color:rgb(233, 105, 0);background-color:rgb(248, 248, 248);">sd</font><font style="color:rgb(52, 73, 94);">的虚拟环境</font>

### <font style="color:rgb(52, 73, 94);">3.5 安装依赖</font>
<font style="color:rgb(52, 73, 94);">按上面的操作把pip替换为国内镜像源后，激活虚拟环境，并安装需要的依赖包</font>

<font style="color:rgb(52, 73, 94);">执行命令：</font>

```shell
# 进入stable-diffusion-webui的文件目录
cd stable-diffusion-webui

# 激活虚拟环境
conda activate sd

# 安装所需依赖
pip3 install -r requirements_versions.txt
```

<font style="color:rgb(52, 73, 94);">这一步如果没任何问题，安装过程算是有惊无险完成了一半。如果有问题，请自行百度谷歌搜索解决，欢迎留言遇到的问题和解法</font>

## <font style="color:rgb(52, 73, 94);">4. 模型安装</font>
### <font style="color:rgb(52, 73, 94);">4.1下载模型</font>
**<font style="color:rgb(52, 73, 94);">官方模型下载（checkpoint模型）</font>**

<font style="color:rgb(52, 73, 94);">下载地址：</font>[https://huggingface.co/CompVis/stable-diffusion-v-1-4-original](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)

<font style="color:rgb(52, 73, 94);">下载</font><font style="color:rgb(52, 73, 94);"> </font><font style="color:rgb(233, 105, 0);background-color:rgb(248, 248, 248);">sd-v1-4.ckpt</font><font style="color:rgb(52, 73, 94);"> </font><font style="color:rgb(52, 73, 94);">或者</font><font style="color:rgb(52, 73, 94);"> </font><font style="color:rgb(233, 105, 0);background-color:rgb(248, 248, 248);">sd-v1-4-full-ema.ckpt</font><font style="color:rgb(52, 73, 94);">。</font>

**<font style="color:rgb(52, 73, 94);">LoRA模型</font>**

<font style="color:rgb(52, 73, 94);">这个应该是大家最喜欢的模型了，懂的都懂。。。</font>

<font style="color:rgb(52, 73, 94);">下载地址：</font>[https://civitai.com/models/6424/chilloutmix](https://civitai.com/models/6424/chilloutmix)

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1689125035120-06c36b90-e139-4de9-947a-29ba8fc1262a.png)

<font style="color:rgb(52, 73, 94);">右上角Download下载，其他模型大家可自行在这个网站上探索，非常的多，这里推荐几个热门的:</font>

[korean-doll-likeness](https://civitai.com/models/7448/korean-doll-likeness)

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1689125034937-985848ab-7882-4352-bd82-76d271ee3db8.png)

### <font style="color:rgb(52, 73, 94);">4.2 安装模型</font>
+ <font style="color:rgb(52, 73, 94);">对于</font>**<font style="color:rgb(52, 73, 94);">checkpoint模型</font>**<font style="color:rgb(52, 73, 94);">，请移动到</font><font style="color:rgb(233, 105, 0);background-color:rgb(248, 248, 248);">stable-diffusion-webui/models/Stable-diffusion</font><font style="color:rgb(52, 73, 94);">⽬录下</font>
+ <font style="color:rgb(52, 73, 94);">对于</font>**<font style="color:rgb(52, 73, 94);">LoRA模型</font>**<font style="color:rgb(52, 73, 94);">，请移动到</font><font style="color:rgb(233, 105, 0);background-color:rgb(248, 248, 248);">stable-diffusion-webui/models/Lora</font><font style="color:rgb(52, 73, 94);">目录下</font>
+ <font style="color:rgb(52, 73, 94);">其他模型按对应的类型移到对应的目录下</font>

## <font style="color:rgb(52, 73, 94);">5. 运行项目</font>
### <font style="color:rgb(52, 73, 94);">5.1 跳过GPU检测</font>
<font style="color:rgb(52, 73, 94);">前面说了，咱们用的是老Mac本了，Intel芯片，显卡也用不了。只能用CPU进行计算，跳过GPU的配置如下：</font>

<font style="color:rgb(52, 73, 94);">执行命令：</font>

```shell
# 打开配置文件
vim ~/.bash_profile

# 把下面两行拷贝进去，保存后source命令使其生效
export COMMANDLINE_ARGS="--lowvram --precision full --no-half --skip-torch-cuda-test"
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### <font style="color:rgb(52, 73, 94);">5.3 项目代码修改</font>
<font style="color:rgb(52, 73, 94);">因为网络访问的问题，我们需要将代码里有些地方进行修改。修改如下：</font>

**<font style="color:rgb(52, 73, 94);">修改lanuch.py文件</font>**

+ <font style="color:rgb(52, 73, 94);">修改def prepare_environment()方法下的两处位置</font>
1. <font style="color:rgb(52, 73, 94);">torch_command中修改</font><font style="color:rgb(233, 105, 0);background-color:rgb(248, 248, 248);">torch==1.13.1 torchvision==0.14.1</font><font style="color:rgb(52, 73, 94);">把原有的版本号数字后面的其他内容去掉</font>
2. <font style="color:rgb(52, 73, 94);">该方法下所有</font><font style="color:rgb(233, 105, 0);background-color:rgb(248, 248, 248);">https://github.com</font><font style="color:rgb(52, 73, 94);">开头的链接，前面都加上</font><font style="color:rgb(233, 105, 0);background-color:rgb(248, 248, 248);">https://ghproxy.com/</font><font style="color:rgb(52, 73, 94);">这样链接就变成如下格式了：</font><font style="color:rgb(233, 105, 0);background-color:rgb(248, 248, 248);">https://ghproxy.com/https://github.com/</font>

<font style="color:rgb(52, 73, 94);">如图所示</font>

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1689125034935-35dd500b-e3ae-422c-8255-2a4ecc8a1594.png)

### <font style="color:rgb(52, 73, 94);">5.3 运行项目</font>
<font style="color:rgb(52, 73, 94);">上面我们使用conda进入了虚拟环境，然后再运行项目即可，执行命令：</font>

```shell
# 激活虚拟环境sd
conda activate sd 

# 进入到stable-diffusion-webui目录下
cd stable-diffusion-webui

# 运行项目
python launch.py
```

<font style="color:rgb(52, 73, 94);">这一步如果人品好的话，第一次就能全部正常运行完，运行完之后，出现</font><font style="color:rgb(233, 105, 0);background-color:rgb(248, 248, 248);">http://127.0.0.1:7860</font><font style="color:rgb(52, 73, 94);">字样说明运行成功了，浏览器打开这个地址就能开始愉快地玩耍了，玩耍方式自行探索哦~</font>

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1689125036354-8694fbce-8440-46d6-9881-af82f5af85d0.png)

## <font style="color:rgb(52, 73, 94);">6.相关问题</font>
### <font style="color:rgb(52, 73, 94);">pip install -r requirements.txt时报错，有一些依赖没有安装上</font>
<font style="color:rgb(52, 73, 94);">解决方法：手动安装一下依赖包</font>

```shell
pip install 缺少的依赖包
```

## <font style="color:rgb(52, 73, 94);">7.模型下载及图片下载</font>
<font style="color:rgb(52, 73, 94);">文章里用到的模型和图片下载方式：公众号</font><font style="color:rgb(233, 105, 0);background-color:rgb(248, 248, 248);">卷福同学</font><font style="color:rgb(52, 73, 94);">内发关键词</font><font style="color:rgb(233, 105, 0);background-color:rgb(248, 248, 248);">AI绘画</font><font style="color:rgb(52, 73, 94);">获取</font>

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1689125037049-33257a30-39ac-4ddb-b5cb-dd0dc202939b.png)

  


