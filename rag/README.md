# CoalQA RAG

<h2 id="1">模块目的：</h2>
 根据用户的问题，检索对应信息以增强回答的专业性, 使CoalQA的回答更加专业可靠。检索内容包括但不限于以下几点：


- 煤矿安全相关的问答对

- 煤矿安全相关的案例

<h2 id="2">技术路线：</h2>
这个RAG系统各部分的技术路线分别是：

- Embedding 模型：BAAI/bge-small-en-v1.5

- LLM基座：InternLM2-Chat-1.8B InternLM2-Chat-7B InternLM2-Chat-20B InternLM2-Math-7B

- 使用BAAI/bge-reranker-large做检索后精排

- 向量数据库：

  - FAISS：是Facebook开源的一个高效的向量相似性搜索库。它为大规模向量检索提供了多种索引结构和搜索算法,能够快速查找与给定向量最相似的向量。FAISS擅长处理超高维度的向量数据,在图像检索、推荐系统等场景有广泛应用。

<h2 id="3">环境搭建：</h2>

```shell
git clone https://github.com/yaosenJ/CoalQA.git
cd rag
pip install -r requirements.txt
```
<!-- **使用指南：** -->
<h2 id="4">使用指南：</h2>
<h3 id="4-1">本地数据集向量化 </h3>

**PDF 数据**  

将需要向量化的pdf数据放到对应的目录下

**JSON 数据** 

构建 QA 对并生成 JSON 文件（单轮/多轮对话），放入到对应目录下

数据格式如下
```shell
# 单轮对话
{
  "conversation": [
      {
          "system": "你是一个煤矿安全领域的知识达人，你会解答很多题。用户给出一个判断题，然后你作出判断，是正确还是错误。",
          "input": "国家对在改善安全生产条件、防止生产安全事故、参加抢险救护等方面取得显著成绩的单位给予奖励;个人给予鼓励。",
          "output": "错误"
      }
  ]
},
# 多轮对话
{
  "conversation": [
      {
          "system": "你是一名煤矿安全领域的知识达人，提供有关煤矿安全规程、事故预防措施和应急响应的知识。",
          "input": "你好！",
          "output": "您好，我是一名煤矿安全领域的知识达人，请问有什么可以帮助您？"
      },
      {
          "input": "2022年9月11日晋能控股集团山西三元煤业股份有限公司下霍煤矿井下无轨胶轮车大巷车场发生了什么事故？",
          "output": "2022年9月11日，晋能控股集团山西三元煤业股份有限公司下霍煤矿井下无轨胶轮车大巷车场发生了一起运输事故，造成1人死亡。"
      },
      {
          "input": "请告诉我这起事故的直接原因是什么？",
          "output": "事故直接原因是井下进行设备换装作业时，机电机运队跟班副队长池晶晶在未采取并确认有效防倒措施的情况下，违章指挥并参与卸车作业，且其本人处在竖装设备倾倒危险区域内，被突然倾倒的端头支架前梁和侧护板砸压受伤致死。"
      },
      {
          "input": "那么，这起事故的间接原因有哪些？",
          "output": "事故间接原因包括现场监督管理严重缺失、规程措施审批把关不严、职工安全风险意识淡薄、上级主体企业责任落实不到位、地方安全监管部门监管不到位。"
      },
      {
          "input": "为了避免类似事故，煤矿企业应该采取哪些措施？",
          "output": "为了避免类似事故，煤矿企业应严格执行安全规程，加强现场安全管理，确保安全技术措施到位，强化安全风险研判，建立完善的安全技术管理体系，落实安全生产主体责任，加强安全教育培训，提高职工安全意识和风险辨识能力，同时加强安全监管，确保监管工作细致认真。"
      },
      {
          "input": "感谢您的详细解答！",
          "output": "不客气，很高兴能帮助到您。如果您还有其他问题，欢迎继续提问。"
      }
  ]
},
```

<h3 id="4-2">构建向量数据库  </h3>
<h4 id="4-2">1. 配置文件修改 </h4>


根据需要改写 config.config 文件：  


```shell
# llm存放位置
model_dir = os.path.join(base_dir, 'model')   

# 向量化模型路径以及模型名称
embedding_path = os.path.join(model_dir, 'embedding_model')         # embedding
embedding_model_name = 'BAAI/bge-small-zh-v1.5'

# 精排模型路径以及模型名称
rerank_path = os.path.join(model_dir, 'rerank_model')  	        	  # embedding
rerank_model_name = 'BAAI/bge-reranker-large'

# 召回documents数量
retrieval_num = 3

# 精排后最终选择留下的documents数量
select_num = 3

prompt_template = """
    你是一个乐于助人的问答代理人。\n
    你的任务是分析并综合检索回来的信息，从而提供有意义且高效的答案。
	{content}
	问题：{query}
"""
```
  
<h4 id="4-3">2. 本地调用 </h4>


运行构建本地知识库脚本  


```shell
python data_generate.py
```
向量化主要步骤如下：

- 加载pdf数据集并提取文本

- 利用RecursiveCharacterTextSplitter按照一定块的大小以及块之间的重叠大小对文本进行分割。

- 加载 BAAI/bge-small-en-v1.5 模型

- 根据文档集构建FAISS索引（即高性能向量数据库）

<h3 id="4-3">相关文本召回与精排</h3>


利用faiss找出与用户输入的问题最相关的文档，然后将召回出来的文本与用户原始输入拼接输入给llm。检索代码如下：


```shell
def get_retrieval_content(self, querys) -> str:
        """
            Input: 用户提问, 是否需要rerank
            ouput: 检索后的内容        
        """
        # print(querys)
        output = []
        content = []
        for query in querys:
            
            documents = self.vectorstores.similarity_search(query, k=self.retrieval_num)
            
            for doc in documents:
                content.append(doc.page_content)
            logger.info(f'Contexts length:{len(content)}')
            if self.rerank_flag:
                model = self.data_processing_obj.load_rerank_model()
                documents = self.data_processing_obj.rerank(model, query, content, self.select_num)

                for doc in documents:
                    output.append(doc)
                logger.info(f'Selected contexts length:{len(output)}')
                logger.info(f'Selected contexts: {output}')
            else:
                logger.info(f'Selected contexts: {content}')
        return output if self.rerank_flag else content
```


<h2 id="5-1">方案细节：</h2>
<h3 id="5-2">RAG具体流程</h3>


- 根据数据集构建 vector DB  

- 对用户输入的问题进行 embedding  

- 基于 embedding 结果在向量数据库中进行检索  

- 对召回数据重排序  

- 依据用户问题和召回数据生成最后的结果  

<h3 id="5-2">后续计划</h3>

- 利用评测框架RAGAS对系统进行评估  

- 构建二分类数据对向量模型以及精排模型进行训练以提升性能  


<h2 id="6-1">相关链接</h2>


BGE Github

- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5): embedding 模型，用于构建 vector DB

- [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large): rerank 模型，用于对检索回来的文章段落重排

InternLM2

- [Chat-1.8B模型](https://huggingface.co/internlm/internlm2-chat-1_8b)

- [Chat-7B模型](https://huggingface.co/internlm/internlm2-chat-7b)

- [Math-7B模型](https://huggingface.co/internlm/internlm2-math-7b)

- [Chat-20B模型](https://huggingface.co/internlm/internlm2-chat-20b)

LangChain

- [文档](https://python.langchain.com/v0.2/docs/introduction/)

- [Github 仓库](https://github.com/langchain-ai/langchain)

FAISS

- [文档](https://faiss.ai/)

- [Github 仓库](https://github.com/facebookresearch/faiss)
