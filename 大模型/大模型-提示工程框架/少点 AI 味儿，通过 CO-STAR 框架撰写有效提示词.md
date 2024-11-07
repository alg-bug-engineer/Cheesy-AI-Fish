# 少点 AI 味儿，通过 CO-STAR 框架撰写有效提示词

有效的提示语（prompt）结构对于从 LLM 那里获得最佳回答至关重要。CO-STAR 框架是新加坡政府科技署数据科学与人工智能团队的心血结晶，是构建提示语（prompt）的便捷模板。其考虑到了影响 LLM 回答的有效性和相关性的所有关键方面，从而有效地优化 LLM 的回答。

![https://oscimg.oschina.net/oscnet/up-ef04858546d4e8dacec91e7baf5f0282c85.png](https://oscimg.oschina.net/oscnet/up-ef04858546d4e8dacec91e7baf5f0282c85.png)

CO-STAR framework — Image by author

其工作原理如下：

- **（C）Context：提供任务的背景信息**
    
    有助于 LLM 理解正在讨论的具体语境，确保其回答是相关的。
    
- **（O）Objective：确定希望 LLM 执行的任务是什么**
    
    明确目标任务有助于 LLM 将其回答重点放在实现某一特定任务上。
    
- **（S）Style：指明希望 LLM 使用的写作风格**
    
    可以设置为某位著名人物的写作风格，也可以是某个专业领域的专家，比如商业分析专家或首席执行官。这样能够引导 LLM 以符合用户需求的方式和用词做出回答。
    
- **（T）Tone：需要确定大模型回答用户提示词的态度**
    
    这样可确保 LLM 的回答与所需的情感或语境相呼应。例如，正式、幽默、感同身受等。
    
- **（A）Audienc：确定回答的对象**
    
    根据回答对象的身份（如某一领域的专家、初学者、儿童等）量身定制 LLM 的回答，确保其在所需的语境中是恰当和可以理解的。
    
- **（R）Response：提供 LLM 回答的格式**
    
    该功能可以让 LLM  以下游任务所需的准确格式输出。例如，列表、JSON（JavaScript Object Notation）、专业报告等。对于大多数以编程方式在下游操作 LLM 回答的 LLM 应用程序来说，JSON 输出格式是最理想的。
    

# **1.1 CO-STAR 的实际应用**

以下是一个 CO-STAR 框架为何实用的实际案例。假设您是一名社交媒体经理（social media manager），需要帮助起草一则 Facebook 帖子，以宣传公司的新产品。

没有使用 CO-STAR 的提示语（prompt）可能如下所示：

> Write a facebook post to advertise my company’s new product. My company’s name is Alpha and the product is called Beta, a new ultra-fast hairdryer.
> 

以下是 GPT-4 的输出结果：

> Introducing the Revolutionary Beta Hairdryer by Alpha!
> 
> - Say goodbye to long, tedious hair drying sessions! Alpha is proud to unveil the Beta, our latest innovation in haircare technology.
> - Ultra-Fast Drying: The Beta is designed to cut your drying time in half, thanks to its advanced, powerful motor. Get ready faster and enjoy more time doing what you love!
> 
> Shop Now: [Link to product page]
> 

虽然内容已经足够了，但这个输出实在过于普通，缺乏引起公司受众产生共鸣所需的，与其他产品的必要细微差异和有针对性的吸引力。

下面将介绍一个 CO-STAR 框架的使用案例，它会提醒您在提示语（prompt）中添加目标任务的其他信息，特别是在之前的 quick prompt 中缺少的 STYLE（风格）、TONE（语气）和 AUDIENCE（受众）等信息：

> # CONTEXT #
> 
> 
> I want to advertise my company’s new product. My company’s name is Alpha and the product is called Beta, which is a new ultra-fast hairdryer.
> 
> # OBJECTIVE #
> 
> Create a Facebook post for me, which aims to get people to click on the product link to purchase it.
> 
> # STYLE #
> 
> Follow the writing style of successful companies that advertise similar products, such as Dyson.
> 
> # TONE #
> 
> Persuasive
> 
> # AUDIENCE #
> 
> My company’s audience profile on Facebook is typically the older generation. Tailor your post to target what this audience typically looks out for in hair products.
> 
> # RESPONSE #
> 
> The Facebook post, kept concise yet impactful.
> 

通过使用 CO-STAR 框架，GPT-4 的回答变得更有针对性、更有效：

> Embrace Effortless Elegance with Alpha Beta — The Ultra-Fast Hairdryer for the Wise Generation.
> 
> 
> Rediscover the joy of simple, effective hair care with Alpha Beta. Our latest innovation is more than just a hairdryer; it’s a promise of swift, gentle, and
> 
> reliable hair styling for those who appreciate the finer things in life.
> 
> - Easy and Intuitive Use: Say goodbye to complicated gadgets. Alpha Beta is
> 
> crafted for comfort and simplicity, perfect for those who value straightforward,
> 
> hassle-free technology.
> 
> - Time-Saving Technology: We understand your time is precious. That’s why Alpha Beta cuts down drying time significantly, giving you more moments to enjoy life’s pleasures.
> 
> Make every day a good hair day with Alpha Beta. Experience the blend of
> 
> sophistication and simplicity today.
> 
> [Your Product Link Here]
> 

CO-STAR 框架指导我们有条不紊地向 LLM 提供相关任务的所有关键信息，确保根据我们的需求提供量身定制的优化回答。