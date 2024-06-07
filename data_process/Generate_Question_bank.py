#coding:utf-8
import csv
import json


csv_filename = '../data/多选题.csv'
json_filename = '../data/multiple_choice.json'

# 读取CSV文件并转换为JSON格式
conversations_list = []
with open(csv_filename, mode='r', encoding='gbk') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        question, answer = row
        conversation = {
            "system": "你是一个煤矿安全领域的知识达人，你会解答很多题。用户给出一个多选题，你能从几个选项中，选择出多个正确选项。",
            "input": question,
            "output": answer
        }
        conversations_list.append({"conversation": [conversation]})

# 创建最终的JSON结构
json_data = conversations_list

# 写入JSON文件
with open(json_filename, mode='w', encoding='utf-8') as jsonfile:
    json.dump(json_data, jsonfile, ensure_ascii=False, indent=4)

print(f"转换完成，JSON文件已保存为 '{json_filename}'")
