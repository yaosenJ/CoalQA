# LMDeploy本地部署

## 0. LMDeploy简介

LMDeploy 由 [MMDeploy](https://github.com/open-mmlab/mmdeploy) 和 [MMRazor](https://github.com/open-mmlab/mmrazor) 团队联合开发，是涵盖了 LLM 任务的全套轻量化、部署和服务解决方案。 这个强大的工具箱提供以下核心功能：

- 高效推理：LMDeploy 通过引入持久批处理（又称连续批处理）、阻塞式 KV 缓存、动态拆分与融合、张量并行、高性能 CUDA 内核等关键功能，将请求吞吐量提高到 vLLM 的 1.8 倍。

- 有效量化：LMDeploy 支持只加权量化和 k/v 量化，4 位推理性能是 FP16 的 2.4 倍。量化质量已通过 OpenCompass 评估确认。

- 轻松分发服务器：利用请求分发服务，LMDeploy 可在多台机器和卡上轻松高效地部署多模型服务。

- 交互式推理模式：通过缓存多轮对话过程中的关注度 k/v，引擎可记住对话历史，从而避免重复处理历史会话。

## 1. 环境安装

pip安装：

```shell
pip install lmdeploy
```

自 v0.3.0 起，默认预编译包在 **CUDA 12** 上编译。不过，如果需要 **CUDA 11+**，可以通过以下方式安装 lmdeploy:

```shell
export LMDEPLOY_VERSION=0.3.0
export PYTHON_VERSION=38
pip install https://github.com/InternLM/lmdeploy/releases/download/v${LMDEPLOY_VERSION}/lmdeploy-${LMDEPLOY_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux2014_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```

## 2. 使用LMDeploy与模型对话

使用LMDeploy与模型进行对话，可以执行如下命令运行下载的1.8B模型

    lmdeploy chat /group_share/internlm2_chat_1_8b_qlora_18000

## 3.LMDeploy模型量化(lite)

#### 3.1 设置最大KV Cache缓存大小

通过 --cache-max-entry-count参数，控制KV缓存占用剩余显存的最大比例为0.5

    lmdeploy chat /group_share/internlm2_chat_1_8b_qlora_18000 --cache-max-entry-count 0.5

#### 3.2 使用W4A16量化

LMDeploy使用AWQ算法，实现模型4bit权重量化。推理引擎TurboMind提供了非常高效的4bit推理cuda kernel，性能是FP16的2.4倍以上。它支持以下NVIDIA显卡：

- 图灵架构（sm75）：20系列、T4
- 安培架构（sm80,sm86）：30系列、A10、A16、A30、A100
- Ada Lovelace架构（sm90）：40 系列

运行前，首先安装一个依赖库。

    pip install einops==0.7.0

仅需执行一条命令，就可以完成模型量化工作。

    lmdeploy lite auto_awq \
       /group_share/internlm2_chat_1_8b_qlora_18000  \
      --calib-dataset 'ptb' \
      --calib-samples 128 \
      --calib-seqlen 1024 \
      --w-bits 4 \
      --w-group-size 128 \
      --work-dir /group_share/internlm2_chat_1_8b_qlora_18000-4bit

## 4. LMDeploy服务(serve)

通过以下lmdeploy命令启动API服务器，推理模型：

    lmdeploy serve api_server \
        /group_share/internlm2_chat_1_8b_qlora_18000-4bit \
        --model-format hf \
        --quant-policy 0 \
        --server-name 0.0.0.0 \
        --server-port 23333 \
        --tp 1

即可以得到FastAPI的接口