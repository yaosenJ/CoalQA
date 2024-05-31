#coding=utf-8
import json
import random

# Load the original data from the provided JSON string
with open('./data/output/Coal_mine_safety_data.json', 'r',encoding='utf-8') as f:
    data = json.load(f)

# Shuffle the data
random.shuffle(data)

# Save the shuffled data to a new JSON file
output_path = './data/output/shuffled_data.json'
with open(output_path, 'w',encoding='utf-8') as f:
    json.dump(data, f, indent=2,ensure_ascii=False)
