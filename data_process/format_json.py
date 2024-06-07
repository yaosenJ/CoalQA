#coding=gbk
import json

def format_json(data):
    formatted_data = json.dumps(data, indent=4, ensure_ascii=False)
    return formatted_data

def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

formatted_data = format_json(read_json("./data/output/Coal_mine_safety_data.json"))

with open("data.json", "w", encoding="utf-8") as f:
    f.write(formatted_data)


