<p align="center">
    <br>
    <img src="https://github.com/yaosenJ/CoalQA/blob/main/imgs/coal_mine_safety.png?raw=true" width="500" height="400"/>
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
    <img src="https://github.com/yaosenJ/CoalQA/blob/main/imgs/%E6%9E%B6%E6%9E%84%E5%9B%BE.png?raw=true" width="920" height="400"/>
    <br>
    <br>
    <img src="https://github.com/yaosenJ/CoalQA/blob/main/imgs/RAG.png?raw=true" width="920" height="400"/>
    <br>
</p>

## 📬 NEWS

- \[**2023/06/03**\] 发布CoalMineLLM_InternLM2-Chat-20B模型到OpenXLab。
- \[**2023/06/01**\] 发布CoalMineLLM_InternLM2-Chat-1_8B、CoalMineLLM_InternLM2-Math-7B模型到OpenXLab。
- \[**2024/05/31**\] 发布CoalMineLLM_InternLM2-Chat-7B模型到OpenXLab。
- \[**2024/05/22**\] 我们启动了煤矿安全领域的大模型项目。

<h2 id="1">🚴 快速使用</h2>

<h3 id="1-1">🛠️ 环境配置</h3>

```shell
git clone https://github.com/yaosenJ/CoalQA.git
cd CoalQA
conda create -n CoalQA python=3.9 -y
conda activate CoalQA
pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

<h3 id="7-3">💡 致谢</h3>

我们非常感谢以下这些开源项目给予我们的帮助：

- [InternLM](https://github.com/InternLM/InternLM)

- [Xtuner](https://github.com/InternLM/xtuner)

- [Imdeploy](https://github.com/InternLM/lmdeploy)

- [InternlM-Tutorial](https://github.com/InternLM/Tutorial)
  
最后感谢上海人工智能实验室推出的书生·浦语大模型实战营，为我们的项目提供宝贵的技术指导和强大的算力支持！
