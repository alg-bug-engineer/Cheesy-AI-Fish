ChatGLM3具备工具调用能力，这意味着它可以利用外部工具来增强自身的功能和性能。通过工具调用，ChatGLM3能够整合各种资源，提供更丰富、更准确的信息。



1. 提升模型性能：通过调用外部工具，ChatGLM3可以弥补自身在某些任务上的不足，提高整体性能。
2. 扩展应用场景：工具使用能力使得ChatGLM3能够适应更多领域和场景的需求，拓宽了其应用范围。
3. 增强实用性：整合各类工具使得ChatGLM3在实际应用中更具实用性和灵活性，能够更好地满足用户需求。



这种工具调用能力使得ChatGLM3在处理复杂任务时更加高效和准确。例如，它可以调用各类API实现联网、接入外部数据等功能，扩展了ChatGLM3的应用场景。



本文介绍下如何用ChatGLM3实现一个本地生活助手的实现，包括查询手机号归属地、天气、生活指数、识别IP等功能，这些以API调用的方式实现。



> 使用的免费API接口：
>
> 百度API商城：[https://apis.baidu.com/](https://apis.baidu.com/)
>



#### 构建 System Prompt
ChatGLM3调用的工具，首先需要生成一个类似yaml格式的描述，通过描述定义了工具的场景、变量含义等，让模型能够“知道”这个工具的作用。如下是使用的三个工具定义：

```python
tools = [
    {
        "name": "search_phone",
        "description": "查询给定手机号的归属地",
        "parameters": {
            "type": "object",
            "properties": {
                "phone": {
                    "description": "需要查询的手机号"
                }
            },
            "required": ['phone']
        }
    },
    {
        "name": "search_weather",
        "description": "查询特定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "description": "需要查询的城市名称"
                }
            },
            "required": ['city']
        }
    },
    {
        "name": "search_index",
        "description": "查询给定城市的生活指数",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "description": "需要查询的城市名"
                }
            },
            "required": ['city']
        }
    }
]
system_info = {"role": "system", "content": "Answer the following questions as best as you can. You have access to the following tools:", "tools": tools}
```

需要注意的点是，在这里定义的工具描述，name是后续写的函数的名；description是对工具的描述，是模型对工具理解的重要信息，描述信息的书写一定不能过于笼统和模糊。parameters是对工具入参的描述，包括必选参数和可选参数以及每个参数的含义。

#### 提出问题
> 注意：目前 ChatGLM3-6B 的工具调用只支持通过 `chat` 方法，不支持  `stream_chat` 方法。
>

以自然语言的方式提出诉求，例如查询xx的天气、生活指数、股票信息等

```python
history = [system_info]
query = "查下北京的天气"
response, history = model.chat(tokenizer, query, history=history)
print(response)
```

这里期望得到的输出为

```json
{'name': 'search_weather', 'parameters': {'city': '北京'}}
```

这表示模型需要调用工具 search_weather，并且需要传入参数 city。

#### 调用工具，生成回复
ChatGLM3第一次是实现对输入的理解和结构化，指出要调用的工具和工具需要的参数，接下来调用工具和传入参数需要自行实现

```python
func = eval(params['name'])
print("the function is:\t", func)
print("params is:\t", params['parameters'])
res = func(**params['parameters'])
```

调用上述工具，获取如下的结果：

```json
the function is:	 <function search_weather at 0x7f22d6a9f2e0>
params is:	 {'city': '北京'}
func result:
{"city": "北京", "info": {"风向": "东风", "天气": "多云", "日期": "11月07日(星期二)"}}
```

将上述结果再次传入大模型

```python
result = json.dumps({"price": 12412}, ensure_ascii=False)
response, history = model.chat(tokenizer, result, history=history, role="observation")
print(response)
```

这里 `role="observation"` 表示输入的是工具调用的返回值而不是用户输入，不能省略。

期望得到的输出为

```plain
根据您的查询，我查询到了北京市的天气情况。今天是11月8日，北京市的天气情况是晴天，风向是东风。希望这些信息能对您有所帮助。
```

这表示本次工具调用已经结束，模型根据返回结果生成回复。对于比较复杂的问题，模型可能需要进行多次工具调用。这时，可以根据返回的 `response` 是 `str` 还是 `dict` 来判断返回的是生成的回复还是工具调用请求。



在上述实现过程中，可以看到，目前ChatGLM3通过对用户输入进行理解后，做出选择何种工具的判断，并将用户的输入转化为工具可接受的输入格式。



在这里有一点需要注意的是，假定工具接受两个参数，以天气为例：日期+城市，用户输入只有一个日期或者城市的时候，大模型是无法判断的，也就是生成的入参是不完整的。



整体来说，目前ChatGLM3向前迈出了一步，能够使用工具，增强应用的边界。但还是存在一些提升的点，例如流程的全自动化、对问题的理解更加准确全面等。



希望以后大模型能够更加智能，甚至自行设计工具，或许那时候，才是真正的通用人工智能时代吧。



