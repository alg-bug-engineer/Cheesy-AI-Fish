## <font style="color:rgb(34, 34, 34);">特点</font>
+ <font style="color:rgb(34, 34, 34);"> </font><font style="color:rgb(34, 34, 34);">自动下载所有模型、LORAs、超网络和嵌入式预览图像</font>
+ <font style="color:rgb(34, 34, 34);"> </font><font style="color:rgb(34, 34, 34);">根据模型哈希自动下载模型，以应用粘贴的生成参数</font>
+ <font style="color:rgb(34, 34, 34);"> </font><font style="color:rgb(34, 34, 34);">元数据中的资源：包括图像中使用的所有资源的SHA256哈希，以便能够自动链接到Civitai上的相应资源</font>
+ <font style="color:rgb(34, 34, 34);"> </font><font style="color:rgb(34, 34, 34);">元数据中的灵活资源命名：在提示中对资源的名称进行哈希处理，以避免出现资源重命名问题，并使提示更易于移植</font>
+ <font style="color:rgb(34, 34, 34);"> </font><font style="color:rgb(34, 34, 34);">Civitai链接：可选的Websocket连接，可在浏览Civitai或其他启用Civitai链接的网站时添加/删除资源等。</font>

## <font style="color:rgb(34, 34, 34);">安装</font>
### <font style="color:rgb(34, 34, 34);">通过扩展UI安装（推荐）</font>
1. <font style="color:rgb(34, 34, 34);">在Automatic1111 SD Web UI中打开扩展选项卡</font>
2. <font style="color:rgb(34, 34, 34);">在扩展选项卡中打开“从URL安装”选项卡</font>
3. <font style="color:rgb(34, 34, 34);">将</font><font style="color:rgb(34, 34, 34);background-color:rgb(238, 238, 238);">https://github.com/civitai/sd_civitai_extension.git</font><font style="color:rgb(34, 34, 34);">粘贴到URL输入框中  
</font>![](https://cdn.nlark.com/yuque/0/2023/png/406504/1689125436232-40af1d23-e321-4d1c-b237-7fe9f20162c0.png)
4. <font style="color:rgb(34, 34, 34);">点击安装并等待完成</font>
5. **<font style="color:rgb(34, 34, 34);">重启Automatic1111</font>**<font style="color:rgb(34, 34, 34);">（重新加载UI不会安装必要的要求）  
</font>![](https://cdn.nlark.com/yuque/0/2023/png/406504/1689125444018-9b290339-ff1b-49f7-a8ff-84473073b9a6.png)

### <font style="color:rgb(34, 34, 34);">手动安装</font>
+ <font style="color:rgb(34, 34, 34);">使用任何方法（zip下载或克隆）下载repo</font>

<font style="color:rgb(214, 50, 0);background-color:rgb(248, 248, 248);">git clone "https://github.com/civitai/sd_civitai_extension.git"</font>

+ <font style="color:rgb(34, 34, 34);">下载repo后，在该位置打开命令提示符</font>

<font style="color:rgb(214, 50, 0);background-color:rgb(248, 248, 248);">cd C:\path\to\sd_civitai_extension</font>

+ <font style="color:rgb(34, 34, 34);">然后运行包含的install.py脚本</font>

