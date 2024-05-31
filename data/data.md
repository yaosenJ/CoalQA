# 数据构建

## 从规程规章制度等文档生成InternLM2模型训练数据格式
生成问答对QA_pair格式
QA_pair:
[
  {
    "input": "请提供新建矿井立井井筒冻结法施工的具体要求。",
    "output": "新建矿井立井井筒冻结法施工需要遵守以下要求：冻结深度必须穿过风化带延深至稳定的基岩10m以上，第一个冻结孔必须全孔取芯，钻孔时必须测定钻孔的方向和偏斜度，偏斜度超过规定时必须及时纠正，冻结管必须采用无缝钢管并焊接或螺纹连接，开始冻结后必须经常观察水文观测孔的水位变化，并在确定冻结壁已交圈后才能进行试挖。"
  },
  ...
  
]

通过如下命令，在当前目录生成CoalQA_data.json
```shell
python Generate_QAdata.py  CoalQA_data.json
```
接上面生成json，处理成InternLM2可训练数据格式，运行下面命令即可
```shell
python format_internlm2.py  
```
## 从整理好题库生成InternLM2模型训练数据格式
[
    {
        "conversation": [
            {
            
                "system": "你是一个煤矿安全领域的知识达人，你会解答很多题。用户给出一个单选题，你能从几个选项中，选择出一个正确选项。",
                "input": "关于《安全生产法》的立法目的，下列表述中不准确的是( ) A.加强安全生产工作  C.保障人民群众生命和财产安全  B.防止和减少生产安全事故 D.提升经济发展速度",
                "output": "D"
                
            }
        ]
    },
    ...
]

```shell
python Generate_Question_bank.py
```


### 补充细节
1.合并两个json文件的脚本：merge_json.py
2.格式化json文本的脚本：format_json.py
3.打乱json中数据顺序的脚本：shuffle.py

