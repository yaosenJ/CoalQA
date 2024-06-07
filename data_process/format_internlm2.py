#coding=utf-8
import json


input_json_filename = '../data/CoalQA_data.json'
output_json_filename = '../data/Xtuner_CoalQA_data.json'

# 读取原始JSON文件
with open(input_json_filename, 'r', encoding='utf-8') as infile:
    data = json.load(infile)

# 转换数据格式
converted_data = [
    {"conversation": [
        {
            "system": "你是一个煤矿安全领域的知识达人，你对相关煤矿安全规章规程制度、技术等文档非常熟悉。请你专业正确地解答用户想问的煤矿安全相关问题。",
            "input": item["input"],
            "output": item["output"]
        }
    ]} for item in data
]

# 写入转换后的JSON文件
with open(output_json_filename, 'w', encoding='utf-8') as outfile:
    json.dump(converted_data, outfile, ensure_ascii=False, indent=4)

print(f"转换完成，JSON文件已保存为 '{output_json_filename}'")
