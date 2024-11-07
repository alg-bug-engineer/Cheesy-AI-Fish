#### <font style="color:rgb(31, 35, 40);">sd-webui-prompt-all-in-one 是一个基于 </font>[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)<font style="color:rgb(31, 35, 40);"> 的扩展，旨在提高提示词/反向提示词输入框的使用体验。它拥有更直观、强大的输入界面功能，它提供了自动翻译、历史记录和收藏等功能，它支持多种语言，满足不同用户的需求。</font>
#### <font style="color:rgb(51, 51, 51);">一、功能介绍</font>
<font style="color:rgb(82, 82, 82);">插件网址：</font><font style="color:rgb(82, 82, 82);"> </font>[https://github.com/Physton/sd-webui-prompt-all-in-one](https://link.uisdc.com/?redirect=https%3A%2F%2Fgithub.com%2FPhyston%2Fsd-webui-prompt-all-in-one)

<font style="color:rgb(82, 82, 82);">Prompt-all-in-one 插件功能主要包括自动中文转英文、一键转英文、快速修改权重、收藏常用提示词等。</font>

<font style="color:rgb(82, 82, 82);">① 中文自动转英文</font>

<font style="color:rgb(82, 82, 82);">我们可以通过插件让中文提示词自动转换为英文。比如在插件的提示框内逐个输入关键词，按 Enter 发送，提示词框内的提示词就会变成对应的英文。</font>

<font style="color:rgb(82, 82, 82);">② 调整提示词位置和权重</font>

<font style="color:rgb(82, 82, 82);">使用 prompt-all-in-one 插件输入提示词后，插件会重新将所有提示词以标签的方式排列出来，我们可以对标签进行如下操作：</font>

1. <font style="color:rgb(82, 82, 82);">用鼠标直接拖动标签，更改顺序</font>
2. <font style="color:rgb(82, 82, 82);">修改提示词权重，省去手工设置的麻烦</font>
3. <font style="color:rgb(82, 82, 82);">直接在标签内修改提示词或者权重，中英文都可以。如果输入的是中文，点击 En 小图标将其转换为英文。</font>

<font style="color:rgb(82, 82, 82);">③ 查看/收藏提示词</font>

<font style="color:rgb(82, 82, 82);">点击“历史记录”小图标，可以查看我们使用过的提示词并进行收藏、复制、使用等操作。提示词收藏后，可以在“收藏列表”中查看，点击“使用”可以将提示词直接填入文本框中，这样我们就就能快速调用常用提示词，非常方便。</font>

#### <font style="color:rgb(51, 51, 51);">二、安装方法</font>
<font style="color:rgb(82, 82, 82);">打开 WebUI，点击“扩展”选项卡，选择“从网址安装”，复制（</font><font style="color:rgb(82, 82, 82);"> </font>[https://github.com/Physton/sd-webui-prompt-all-in-one.git](https://link.uisdc.com/?redirect=https%3A%2F%2Fgithub.com%2FPhyston%2Fsd-webui-prompt-all-in-one.git)<font style="color:rgb(82, 82, 82);"> </font><font style="color:rgb(82, 82, 82);">），粘贴在第一行的“拓展的 git 仓库网址”中。点击“安装”按钮，等待十几秒后，在下方看到一行小字“Installed into stable-diffusion-webui\extensions\sd-webui-controlnet. Use Installed tab to restart”，表示安装成功。</font>

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1689127717173-448718e7-048e-4da3-87ce-5eea27386010.png)

<font style="color:rgb(82, 82, 82);">点击左侧的“已安装”选项卡，单击“检查更新”，等待进度条完成；然后单击“应用并重新启动 UI”；最后完全关闭 WebUI 程序，重新启动进入（也可以重启电脑），我们就可以在 WebUI 主界面中看到 Prompt-all-in-one 的选项。</font>

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1689127717181-0b169d73-e87c-4cf7-8eb2-d7b0dcd9fea7.png)

<font style="color:rgb(153, 153, 153);">安装成功后的 WebUI 界面</font>

<font style="color:rgb(82, 82, 82);">安装成功后，我们还需要进行三个简单的设置。</font>

<font style="color:rgb(82, 82, 82);">① 设置语言类型</font>

<font style="color:rgb(82, 82, 82);">这个非常简单，直接点击地球小图标，将语言类型设置为“简体中文”，这样我们在输入中文简体的时候，插件就会自动把更改为英文。</font>

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1689127717258-529abdc9-dab5-43dd-b372-37b80130bdeb.png)

<font style="color:rgb(82, 82, 82);">② 设置翻译API</font>

<font style="color:rgb(82, 82, 82);">点击设置小图标，选择第一个 API 图标，然后在弹出的窗口内设置翻译接口。插件提供看非常多免费的 API，包括百度、有道、彩云翻译等，大家可以随便选择一个带[Free]标志的就可以了。选择完毕后记得点击下方的“测试”按钮，检查刚刚选择的借口能否正常使用，如果可以，就会显示出文本翻译结果；如果不行，就换另一个免费的翻译接口。测试成功后完，点击保存即可。</font>

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1689127717314-16ab9b8b-c4b7-412c-88cb-ce8857df7594.png)

<font style="color:rgb(82, 82, 82);">③ 开启一键翻译</font>

<font style="color:rgb(82, 82, 82);">点击设置小图标，如下图进行勾选，正负提示词都需要进行同样设置。到这一步插件就安装完成后，大家可以按照前面的功能呢过介绍进行操作。</font>

![](https://cdn.nlark.com/yuque/0/2023/png/406504/1689127717098-3e16d671-7288-492e-9406-e5fc67f240ea.png)

