# 如何使用 Dreambooth 将任何东西放入稳定扩散（Colab 笔记本）
![](https://cdn.nlark.com/yuque/0/2023/png/406504/1690593456575-0b07d16d-3b8f-4db9-bd63-ce58bb5a47b4.png)

Dreambooth 是一种将任何东西（您所爱的人、您的狗、您最喜欢的玩具）放入稳定扩散模型的方法。将介绍 Dreambooth 是什么、它是如何工作的以及如何进行训练。



按照分步指南准备训练图像，并使用我们简单的一键式 Colab 笔记本进行 dreambooth 训练 - 无需编码！



## 什么是Dreambooth？
[Dreambooth](https://dreambooth.github.io/)由 Google 研究团队于 2022 年发布，是一种通过向模型注入自定义主题来微调扩散模型（如稳定扩散）的技术。

为什么叫Dreambooth？据谷歌研究团队称，它就像一个照相亭，但一旦拍摄到主题，它就可以合成到您的梦想带您去的任何地方。

下面是研究文章中的一个例子。仅使用特定狗（我们称之为Devora）的 3 张图像作为输入，dreamboothed 模型就可以在不同的环境中生成 Devora 的图像。

![只需 3 个训练图像，Dreambooth 即可将自定义主题无缝注入扩散模型](https://cdn.nlark.com/yuque/0/2023/png/406504/1690593455320-dd98c954-8bcd-4959-a53e-08a4b6cf0bc4.png)

### Dreambooth如何运作？
可能会问，为什么不能简单地使用这些图像通过额外的步骤来训练模型？问题是，众所周知，这样做会因过度拟合（因为数据集非常小）和[语言漂移](https://arxiv.org/abs/1909.04499)而导致灾难性的失败。

Dreambooth 通过以下方式解决了这些问题

1. 对新主题使用一个罕见的单词（请注意，这里为狗使用了一个罕见的名字Devora），这样它一开始在模型中就没有太多意义。
2. 类的预先保留：为了保留类（上例中的狗）的含义，模型以注入主体（Devora）的方式进行微调，同时生成类（狗）的图像。保存下来。

还有另一种类似的技术称为[文本反转](https://textual-inversion.github.io/)。不同之处在于，Dreambooth 对整个模型进行了微调，而文本反转则注入了一个新词，而不是重复使用生僻词，并且仅对模型的文本嵌入部分进行了微调。

### 训练 Dreambooth 需要什么？
你需要三样东西

1. 一些自定义图像
2. 唯一标识符
3. 一个类名

在上面的例子中。唯一标识符是Devora。班级名称是狗。

然后你需要构建你的实例提示：

> a photo of [unique identifier] [class name]
>

还有类提示：

> a photo of [class name]
>

在上面的例子中，实例提示符是

> a photo of Devora dog
>

由于 Devora 是一只狗，所以类提示是

> a photo of a dog
>

## 分步指南
### 获取训练图像
与任何机器学习任务一样，高质量的训练数据是成功的最重要因素。为你的自定义主题拍摄 3-10 张照片。照片应该从不同的角度拍摄。拍摄对象还应该处于多种背景中，以便模型可以将拍摄对象与背景区分开来。在教程中使用这个玩具。

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1690593454663-b7a1ddd6-06f6-4435-8f00-52baa13dc90f.png)![](https://cdn.nlark.com/yuque/0/2023/png/406504/1690593454513-d191cba5-dfb5-4be2-a583-942e1846bc10.png)![](https://cdn.nlark.com/yuque/0/2023/png/406504/1690593455296-9d3394a4-5408-49e5-8249-dd5d6dcfd296.png)

### 调整图像大小
为了在训练中使用图像，您首先需要将它们的大小调整为 512×512 像素，以便使用 v1 模型进行训练。

[BIRME](https://www.birme.net/?target_width=512&target_height=512)是一个调整图像大小的便捷网站。

1. 将您的图像拖放到 BIRME 页面。
2. 调整每张图像的画布，使其充分显示主题。
3. 确保宽度和高度均为 512 像素。
4. 按“保存文件”将调整大小的图像保存到您的计算机。

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1690593455815-ce00f91c-bd73-4960-9314-1f57db868d83.png)

或者，如果只想完成教程，可以下载我调整大小的图像。[dreambooth_training_images](https://stable-diffusion-art.com/wp-content/uploads/2022/12/dreambooth_training_images.zip)[下载](https://stable-diffusion-art.com/wp-content/uploads/2022/12/dreambooth_training_images.zip)

### 训练
我建议使用 Google Colab 进行训练，因为它可以省去您设置的麻烦。以下笔记本是从 Shivam Shrirao 的存储库修改而来的，但变得更加用户友好。如果您喜欢其他设置，请按照存储库的说明进行操作。

整个训练时间约为30分钟。如果您不经常使用 Google Colab，您可能可以在不断开连接的情况下完成训练。购买一些计算积分以避免断开连接的挫败感。截至 2022 年 12 月，10 美元即可获得 50 小时，因此成本并不高。

笔记本会将模型保存到您的[Google Drive](https://drive.google.com/drive/my-drive)中。如果您选择（推荐），请确保您至少有 2GB fp16；如果您不选择，请确保您至少有 4GB。

1. 打开 Colab 笔记本。
2. [如果您想从Stable Diffusion v1.5](https://stable-diffusion-art.com/models/#Stable_diffusion_v15)模型（推荐）进行训练，则无需更改 MODEL_NAME 。
3. 放入实例提示符和类提示符。对于我的图像，我将我的玩具兔子命名为 zwx，因此我的实例提示是“zwx 玩具的照片”，我的类提示是“玩具的照片”。

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1690593456473-cb333ddb-b42c-4ab1-b5c4-fa429ae5d0d1.png)

4. 单击单元格左侧的“训练”按钮 ( ▶️ ) 开始处理。

5. 授予访问 Google 云端硬盘的权限。目前，除了将模型文件保存到 Google 云端硬盘之外，没有简单的方法可以下载该模型文件。

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1690593456830-65836436-40ca-4761-9587-d12155a16930.png)

6. 按选择文件上传调整大小的图像。

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1690593457375-b24e03cf-0667-4dec-bab5-3836fb656fcd.png)

7. 完成训练大约需要30分钟。完成后，您应该会看到新模型生成的一些示例图像。

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1690593458030-4650afe5-e68b-4a70-ab51-6bff000f810c.png)

8. 您的自定义模型将保存在您的[Google Drive](https://drive.google.com/drive/my-drive)文件夹下Dreambooth_model。下载模型检查点文件并将其[安装](https://stable-diffusion-art.com/models/#How_to_install_and_use_a_model)在您喜欢的 GUI 中。

就是这样！

### 测试模型
您还可以使用笔记本的第二个单元来测试模型的使用情况。

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1690593457960-b30ff32c-1da6-402a-80ea-0468d1e89de2.png)

使用提示

> 梵高风格zwx油画
>

对于新训练的模型，得到的结果感到：

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1690593458469-0d1c6f02-c6da-4d6e-adbf-586a34f0569c.png)![](https://cdn.nlark.com/yuque/0/2023/png/406504/1690593458875-e8506787-30ea-4e17-ba36-e9ad7be1f8fc.png)

## 使用模型
您可以在 AUTOMATIC1111 GUI 中使用模型检查点文件。它是一个免费且功能齐全的 GUI，您可以在 Windows 和 Mac 上安装，或在[Google Colab](https://stable-diffusion-art.com/automatic1111-colab/)上运行。

