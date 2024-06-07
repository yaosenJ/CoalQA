# 模型微调过程说明文档

## 概览

本文档提供了使用 XTuner 工具进行模型微调过程的详细指南。该过程包括转换、合并、训练以及为不同规模的模型（1.8B 和 20B）设置网络演示。

## 要求

- XTuner
- DeepSpeed
- Huggingface Transformers
- 具备 SSH 和 Git 的使用权限

### 环境安装

```
# 如果你是在 InternStudio 平台，则从本地 clone 一个已有 pytorch 的环境：
# pytorch    2.0.1   py3.10_cuda11.7_cudnn8.5.0_0

studio-conda xtuner0.1.17
# 如果你是在其他平台：
# conda create --name xtuner0.1.17 python=3.10 -y

# 激活环境
conda activate xtuner0.1.17
# 进入家目录 （~的意思是 “当前用户的home路径”）
cd ~
# 创建版本文件夹并进入，以跟随本教程
mkdir -p /root/xtuner0117 && cd /root/xtuner0117

# 拉取 0.1.17 的版本源码
git clone -b v0.1.17  https://github.com/InternLM/xtuner
# 无法访问github的用户请从 gitee 拉取:
# git clone -b v0.1.15 https://gitee.com/Internlm/xtuner

# 进入源码目录
cd /root/xtuner0117/xtuner

# 从源码安装 XTuner
pip install -e '.[all]'
```



## 1. 1.8B 模型训练

### 1.1 数据准备

```
# 在ft这个文件夹里再创建一个存放数据的data文件夹,存储数据
mkdir -p /root/ft/data && cd /root/ft/data
```

### 1.2 准备模型

```
# 创建目标文件夹，确保它存在。
# -p选项意味着如果上级目录不存在也会一并创建，且如果目标文件夹已存在则不会报错。
mkdir -p /root/ft/model

# 复制内容到目标文件夹。-r选项表示递归复制整个文件夹。
cp -r /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b/* /root/ft/model/
```

如果是需要自己下载，可以使用transformers库

```
from transformers import AutoModel

# 指定模型名称
model_name = 'internlm/internlm2-chat-1_8b'

# 加载模型
model = AutoModel.from_pretrained(model_name)

# 指定保存模型的目录
model_save_path = '/root/ft/model'

# 保存模型
model.save_pretrained(model_save_path)

```

将这段代码保存为 `download_model.py`，然后在命令行中运行这个脚本：

```
python download_model.py
```

这个脚本会自动下载模型并将其保存到指定的 `/root/ft/model` 目录中。

### 1.3 下载配置文件

```
# XTuner 提供多个开箱即用的配置文件，用户可以通过下列命令查看：
# 列出所有内置配置文件
# xtuner list-cfg

# 假如我们想找到 internlm2-1.8b 模型里支持的配置文件
xtuner list-cfg -p internlm2_1_8b

# 创建一个存放 config 文件的文件夹
mkdir -p /root/ft/config

# 使用 XTuner 中的 copy-cfg 功能将 config 文件复制到指定的位置
xtuner copy-cfg internlm2_1_8b_qlora_alpaca_e3 /root/ft/config
```

### 1.4 修改配置参数

```
# 修改模型地址（在第27行的位置）
- pretrained_model_name_or_path = 'internlm/internlm2-1_8b'
+ pretrained_model_name_or_path = '/root/ft/model'

# 修改数据集地址为本地的json文件地址（在第31行的位置）
- alpaca_en_path = 'tatsu-lab/alpaca'
+ alpaca_en_path = '/root/ft/data/personal_assistant.json'

# 修改max_length来降低显存的消耗（在第33行的位置）
- max_length = 2048
+ max_length = 1024

# 减少训练的轮数（在第44行的位置）
- max_epochs = 3
+ max_epochs = 2

# 增加保存权重文件的总数（在第54行的位置）
- save_total_limit = 2
+ save_total_limit = 3

# 修改每多少轮进行一次评估（在第57行的位置）
- evaluation_freq = 500
+ evaluation_freq = 300

# 修改具体评估的问题（在第59到61行的位置）

# 把 OpenAI 格式的 map_fn 载入进来（在第15行的位置）
- from xtuner.dataset.map_fns import alpaca_map_fn, template_map_fn_factory
+ from xtuner.dataset.map_fns import openai_map_fn, template_map_fn_factory

# 将原本是 alpaca 的地址改为是 json 文件的地址（在第102行的位置）
- dataset=dict(type=load_dataset, path=alpaca_en_path),
+ dataset=dict(type=load_dataset, path='json', data_files=dict(train=alpaca_en_path)),

# 将 dataset_map_fn 改为通用的 OpenAI 数据集格式（在第105行的位置）
- dataset_map_fn=alpaca_map_fn,
+ dataset_map_fn=None,
```

### 1.5 模型训练

```
# 指定保存路径
xtuner train /root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py --work-dir /root/ft/train

# 使用 deepspeed 来加速训练
xtuner train /root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py --work-dir /root/ft/train_deepspeed --deepspeed deepspeed_zero2
```

### 1.6 转换到 Huggingface 格式

1. **创建目录**：为转换后的 Huggingface 模型创建一个存储目录：
   ```bash
   mkdir -p /root/ft/huggingface/i8000
   ```

2. **模型转换**：使用提供的配置和权重文件进行模型转换：
   ```bash
   xtuner convert pth_to_hf /root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py /root/ft/train_deepspeed/iter_18000.pth /root/ft/huggingface/i8000 --fp32
   ```

3. **合并模型**：合并模型并解决依赖关系：
   ```bash
   mkdir -p /root/ft/final_model_8000
   export MKL_SERVICE_FORCE_INTEL=1
   xtuner convert merge /root/ft/model /root/ft/huggingface/1i8000 /root/ft/final_model_18000
   ```

4. **测试模型**：通过启动对话来测试模型：
   ```bash
   xtuner chat /root/ft/final_model_18000 --prompt-template internlm2_chat
   ```

### 1.7 模型续训

```bash
xtuner train /root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py --work-dir /root/ft/train_deepspeed --resume /root/ft/train_deepspeed/iter_8500.pth  --deepspeed deepspeed_zero1
```

### 1.8 网络演示设置

1. **准备环境**：
   ```bash
   mkdir -p /root/ft/web_demo && cd /root/ft/web_demo
   git clone https://github.com/InternLM/InternLM.git
   cd /root/ft/web_demo/InternLM
   ```

2. **运行演示** 使用 Streamlit：
   ```bash
   streamlit run /root/ft/web_demo/InternLM/chat/web_demo.py --server.address 127.0.0.1 --server.port 6006
   ```

3. **通过 SSH 隧道访问演示**：
   ```bash
   ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 开发机端口号
   ```

## 2. 20B 模型训练

与1.8B模型训练过程类似，20B模型训练涉及到为配置、数据和最终模型创建相应的目录。此外，这一过程还包括使用多个GPU进行模型训练，并将模型转换为Huggingface格式。

### 2.1 数据准备

为大规模的20B模型训练准备数据。

```bash
# 创建一个专用于存放20B模型数据的目录
mkdir -p /root/ft20b/data && cd /root/ft20b/data
```

### 2.2 准备模型

准备模型包括创建目标文件夹并将预训练的20B模型复制到指定位置。

```bash
# 创建一个目录用来存放20B模型文件
mkdir -p /root/ft20b/model

# 将预训练的模型复制到新创建的目录中
cp -r /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-20b/* /root/ft20b/model/
```

### 2.3 下载配置文件

下载并准备20B模型的配置文件，以便进行训练。

```bash
# 列出所有支持20B模型的配置文件
xtuner list-cfg -p internlm2_20b

# 创建一个目录用于存放20B模型的配置文件
mkdir -p /root/ft20b/config

# 复制所需的配置文件到新创建的目录中
xtuner copy-cfg internlm2_20b_qlora_alpaca_e3 /root/ft20b/config
```

### 2.4 修改配置参数

根据训练需求调整配置文件，以优化20B模型的训练。

```bash
# 修改模型路径和数据集路径等关键参数以适配20B模型
- pretrained_model_name_or_path = 'internlm/internlm2-20b'
+ pretrained_model_name_or_path = '/root/ft20b/model'

- alpaca_en_path = 'tatsu-lab/alpaca'
+ alpaca_en_path = '/root/ft20b/data/specific_dataset.json'

- max_length = 2048
+ max_length = 1024

- max_epochs = 3
+ max_epochs = 2

- save_total_limit = 2
+ save_total_limit = 3

- evaluation_freq = 500
+ evaluation_freq = 300
```

### 2.5 模型训练

使用DeepSpeed和多GPU配置来加速20B模型的训练过程。

```bash
# 指定保存路径并开始训练
xtuner train /root/ft20b/config/internlm2_20b_qlora_alpaca_e3_copy.py --work-dir /root/ft20b/train_deepspeed --deepspeed deepspeed_zero2
```

### 2.6 转换到 Huggingface 格式

为转换后的Huggingface模型创建目录并执行转换。

```bash
# 创建一个目录用于存放转换后的Huggingface模型
mkdir -p /root/ft20b/huggingface

# 执行模型转换
xtuner convert pth_to_hf /root/ft20b/config/internlm2_20b_qlora_alpaca_e3_copy.py /root/ft20b/train_deepspeed/iter_2600.pth /root/ft20b/huggingface
```

### 2.7 模型合并

合并转换后的模型并解决依赖关系。

```bash
# 创建一个名为final_model的目录以存储合并后的模型文件
mkdir -p /root/ft20b/final_model

# 合并模型
xtuner convert merge /root/ft20b/model /root/ft20b/huggingface /root/ft20b/final_model
```

### 2.8 测试模型

通过启动对话来测试合并后的模型。

```bash
# 启动与模型的对话测试
xtuner chat /root/ft20b/final_model --prompt-template

 internlm2_chat
```

这一部分提供了详细的指导，确保20B模型的训练过程得到妥善管理和执行。

## 3. 微调20b配置样例

```
max_length = 4096
pack_to_max_length = True

# parallel
sequence_parallel_size = 1

# Scheduler & Optimizer
batch_size = 4  # per_device
accumulative_counts = 16
accumulative_counts *= sequence_parallel_size
dataloader_num_workers = 0
max_epochs = 50

=》

+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.03              Driver Version: 535.54.03    CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-80GB          On  | 00000000:89:00.0 Off |                    0 |
| N/A   65C    P0             334W / 400W |  59119MiB / 81920MiB |    100%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-SXM4-80GB          On  | 00000000:B3:00.0 Off |                    0 |
| N/A   66C    P0             358W / 400W |  59119MiB / 81920MiB |    100%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
+---------------------------------------------------------------------------------------+
```

## 4. 其他注意事项

### 4.1. 单卡训完的，不可以在双卡上续训

原因是：

**问题的根源**：尝试加载的模型检查点是在数据并行（DP）世界大小为1（即单个GPU或单个训练进程）的环境中保存的。但当前尝试恢复训练的环境具有数据并行世界大小为2（即两个GPU或两个训练进程）。

**ZeRO的限制**：DeepSpeed的ZeRO优化器分区（ZeRO-Optimizer State Partitioning）依赖于特定的世界大小配置，并且目前不支持自动调整新的世界大小。换句话说，如果你在一个GPU上训练并保存了检查点，那么在加载这个检查点进行恢复训练时，你必须在相同数量的GPU上进行。

- **性能最优配置**包括设置最大序列长度、批量大小及其他 DeepSpeed 特定设置。

## 结语

本文档作为有效导航模型微调过程的指南，解决了利用 XTuner 和 DeepSpeed 进行微调的技术和配置相关的重要方面。
