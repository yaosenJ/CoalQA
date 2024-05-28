# internlm2微调

## 环境配置

创建环境
```shell
conda create -n internlm2 python=3.10
conda activate internlm2
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

环境包的安装

```shell
cd ~
git clone -b v0.1.18 https://github.com/InternLM/XTuner
cd XTuner
pip install [-e .]
```

##  下载本项目仓库

```shell
git clone https://github.com/yaosenJ/CoalQA.git
```

## 下载模型

进入finetune目录

```shell
cd CoalQA/finetune
```

执行如下命令，下载internlm2-chat-7b模型参数文件：
```shell
python download_model.py
```
## 开始模型微调
