<p align="center">
    <br>
    <img src="https://github.com/yaosenJ/CoalQA/blob/main/imgs/coal_mine_safety.png?raw=true" width="300" height="300"/>
    <br>
</p>

# 煤矿安全大模型————矿途智护者

**本项目简介：**

近年来，国家对煤矿安全生产的重视程度不断提升。为了确保煤矿作业的安全，提高从业人员的安全知识水平显得尤为重要。鉴于此，目前迫切需要一个高效、集成化的解决方案，该方案能够整合煤矿安全相关的各类知识，为煤矿企业负责人、安全管理人员、矿工提供一个精确、迅速的信息查询、学习与决策支持平台。
为实现这一目标，我们利用包括煤矿历史事故案例、事故处理报告、安全操作规程、规章制度、技术文档以及煤矿从业人员入职考试题库等在内的丰富数据资源，通过微调InternLM2模型，构建出一个专门针对煤矿事故和煤矿安全知识智能问答的煤矿安全大模型。

**本项目的特点如下：**

- 支持煤矿安全领域常规题型解答，如：单选题、多选题、判断题、填空题等 （针对煤矿主要负责人及安管人员、煤矿各种作业人员）
- 支持针对安全规程规章制度、技术等文档内容回答（如《中华人民共和国矿山安全法》、《煤矿建设安全规程》）
- 支持煤矿历史事故案例，事故处理报告查询，提供事故原因详细分析、事故预防措施以及应急响应知识

 类别     | 底座   | 名称                      | 版本 | 下载链接                                                     |微调方法|
| -------- | ------ | ------------------------- | ---- | ------------------------------------------------------------ |---------|
| 对话模型 | InternLM2-Chat-1_8B|CoalMineLLM_InternLM2-Chat-1_8B    | V1.0 | [OpenXLab](https://openxlab.org.cn/models/detail/viper/CoalMineLLM_InternLM2-Chat-1_8B)|QLora|
| 对话模型 | InternLM2-Chat-7B  |CoalMineLLM_InternLM2-Chat-7B      | V1.0 | [OpenXLab](https://openxlab.org.cn/models/detail/viper/CoalMineLLM_InternLM2-Chat-7B)|QLora|
| 对话模型 | InternLM2-Math-7B  |CoalMineLLM_InternLM2-Math-7B      | V1.0 | [OpenXLab](https://openxlab.org.cn/models/detail/viper/CoalMineLLM_InternLM2-Math-7B)|QLora|
| 对话模型 | InternLM2-Chat-20B |CoalMineLLM_InternLM2-Chat-20B     | V1.0 | [OpenXLab](https://openxlab.org.cn/models/detail/viper/CoalMineLLM_InternLM2-Chat-20B)|QLora|

## 📍 架构图

<p align="center">
    <br>
    <img src="https://github.com/yaosenJ/CoalQA/blob/main/imgs/%E6%9E%B6%E6%9E%84%E5%9B%BE3.png?raw=true" width="920" height="400"/>
    <br>
    <br>
    <img src="https://github.com/yaosenJ/CoalQA/blob/main/imgs/RAG.png?raw=true" width="920" height="400"/>
    <br>
</p>

## 📬 NEWS
- \[**2024/06/05**\] CoalMineLLM_InternLM2-Chat-1_8V1.0版已部署上线[https://openxlab.org.cn/apps/detail/milowang/CoalQAv1](https://openxlab.org.cn/apps/detail/milowang/CoalQAv1)。
- \[**2024/06/03**\] 发布CoalMineLLM_InternLM2-Chat-20B模型到OpenXLab。
- \[**2024/06/01**\] 发布CoalMineLLM_InternLM2-Chat-1_8B、CoalMineLLM_InternLM2-Math-7B模型到OpenXLab。
- \[**2024/05/31**\] 发布CoalMineLLM_InternLM2-Chat-7B模型到OpenXLab。
- \[**2024/05/22**\] 我们启动了煤矿安全领域的大模型项目。

## 🗂️ 目录

- [🚴 快速开始](#1-快速使用)
  - [本地Demo部署](#11-本地Demo部署)
  - [在线体验](#12-在线体验)
- [详细指南](#2-详细指南)
  - [环境配置](#21-环境配置)
  - [数据构造](#22-数据构造)
  - [模型微调](#23-模型微调)
  - [RAG](#24-RAG)
  - [部署](#25-部署)
- [案例展示](#3-案例展示)
- [人员贡献](#4-人员贡献)
- [💡 致谢](#5-致谢)

<h2 id="1">🚴 快速使用 </h2>

<h3 id="1-1">本地Demo部署 </h3>

```shell
git clone https://github.com/yaosenJ/CoalQA.git
cd CoalQA
conda create -n CoalQA python=3.10.0 -y
conda activate CoalQA
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
cd web_app
streamlit run streamlit_app.py --server.address=127.0.0.1 --server.port 6006
```

<h3 id="1-2">在线体验 </h3>

CoalMineLLM_InternLM2-Chat-1_8版体验地址：[https://openxlab.org.cn/apps/detail/milowang/CoalQAv1](https://openxlab.org.cn/apps/detail/milowang/CoalQAv1)

<h2 id="2"> 详细指南 </h2>

<h3 id="2-1"> 环境配置 </h3>

```shell
git clone https://github.com/yaosenJ/CoalQA.git
cd CoalQA
conda create -n CoalQA python=3.10.0 -y
conda activate CoalQA
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

<h3 id="2-2"> 数据构造 </h3>

- 请查看[数据构造指南](https://github.com/yaosenJ/CoalQA/tree/main/data_process#readme)

- 相关数据请见[CoalQA/data](https://github.com/yaosenJ/CoalQA/tree/main/data)

<h3 id="2-3"> 模型微调 </h3>

- 详细查看模型微调，请查阅[模型微调指南](https://github.com/yaosenJ/CoalQA/blob/main/finetune/README.md)

<h3 id="2-4"> RAG </h3>

- 检索增强生成RAG相关细节，详见：[RAG详细指南](https://github.com/yaosenJ/CoalQA/tree/main/rag#readme)
- 使用Neo4j和Langchain集成非结构化和图知识增强煤矿事故QA，见[CoalMineLLM-InternLM2-Chat-1_8B版](https://github.com/yaosenJ/CoalQA/blob/main/demo/CoalMineLLM-InternLM2_Chat-1_8B-integrated-qa-neo4j-langchain.ipynb)

<h3 id="2-5"> 部署 </h3>

- 本地部署：详见[本地部署说明](web_app/README.md)
- openxlab部署：详见[openxlab部署说明](web_app/publish/README.md)
- 基于[LMDeploy](https://github.com/InternLM/lmdeploy/)的量化部署：详见[deploy](web_app/lmdeploy.md)

<h2 id="3"> 案例展示 </h2>
<p align="center">
    <br>
    <img src="https://github.com/yaosenJ/CoalQA/blob/main/imgs/%E8%87%AA%E6%88%91%E8%AE%A4%E7%9F%A5.png?raw=true" width="800" height="400"/>
    <br>
    <br>
    <img src="https://github.com/yaosenJ/CoalQA/blob/main/imgs/%E7%AD%94%E9%A2%98.png?raw=true" width="800" height="400"/>
    <br>
    <br>
    <img src="https://github.com/yaosenJ/CoalQA/blob/main/imgs/%E7%85%A4%E7%9F%BF%E5%AE%89%E5%85%A8%E7%AD%94%E9%A2%98.png?raw=true" width="800" height="500"/>
    <br>
     <br>
    <img src="https://github.com/yaosenJ/CoalQA/blob/main/imgs/%E4%BA%8B%E6%95%85%E6%9F%A5%E8%AF%A2.png?raw=true" width="800" height="400"/>
    <br>
</p>


<h2 id="4"> 人员贡献 </h2>

[yaosenJ](https://github.com/yaosenJ): 项目发起人，负责数据构造、模型微调、文档整理

[kaiwang0112006](https://github.com/kaiwang0112006): 负责数据构造、模型部署、测试

[qzd-1](https://github.com/qzd-1): 负责RAG

[Volta-lemon](https://github.com/Volta-lemon): 负责模型微调、测试

[Hao813](https://github.com/Hao813): 负责数据收集、架构图绘制


<h2 id="5">💡 致谢</h2>


我们非常感谢以下这些开源项目给予我们的帮助：

- [InternLM](https://github.com/InternLM/InternLM)

- [Xtuner](https://github.com/InternLM/xtuner)

- [Imdeploy](https://github.com/InternLM/lmdeploy)

- [InternlM-Tutorial](https://github.com/InternLM/Tutorial)
  
最后感谢上海人工智能实验室推出的书生·浦语大模型实战营，为我们的项目提供宝贵的技术指导和强大的算力支持！
