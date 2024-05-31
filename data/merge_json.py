#coding=utf-8
import json

def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def merge_json(file1,file2):
    data1 = read_json(file1)
    data2 = read_json(file2)

    merged_data = data1 + data2
    return merged_data

# 示例
merged_data = merge_json("./data/output/Coal_mine_safety_data.json", "./data/output/output.json")

with open("./data/output/Coal_mine_safety_data.json", "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)
