#coding:utf-8
import sys
import random
from zhipuai import ZhipuAI
client = ZhipuAI(api_key="your api_key")

with open('data/19-中华人民共和国矿山安全法.txt', 'r', encoding='utf-8') as f:
    content = f.read()

def return_random_prompt():
  system_prompt = "根据下面提供有关煤矿安全领域文本，请你仔细通读全文，你需要依据该文本：\n\n######\n{}######\n尽可能给出多样化的问题和对应的回答。我们将用于人工评估GLM-4模型对问答对数据的完成情况。要求:\n".format(content)
  system_prompt += "1. 生成问题有价值且遵守该文本信息，回答准确专业。\n"
  system_prompt += "2. 问题多样化，同个问题可以换成不同表述方式，但意思保持不变。\n"
  system_prompt += "3. 为问题生成一个适当且涉及真实情况的<input>，不应该只包含简单的占位符。<input>应提供实质性的内容，具有挑战性。字数不超过" + str(random.randint(80, 120)) + "字。\n"
  system_prompt += "4. <output>应该是对问题的适当且真实的回答，不能只回复答应或拒绝请求。如果需要额外信息才能回复时，请努力预测用户意图并尝试回复。<output>的内容应少于" + str(random.randint(128, 512)) + "字。\n\n"
  system_prompt += "请给出满足条件的20条JSON格式数据，并存储在一个列表中，便于整理使用\n"
  return system_prompt


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python Generate_QAdata.py <output_file>")
    exit(1)

  output_file = open(sys.argv[1], 'w')

  MAX_EPOCHS = 1    # number of data to generate (each prompt contains 20 JSON-formatted data)

  for k in range(MAX_EPOCHS):
      response = client.chat.completions.create(
          model="glm-4",
          messages=[
              {
                  "role": "user",
                  "content": return_random_prompt()
              }
          ],
          top_p=0.7,
          temperature=0.9,
          stream=False,
          max_tokens=2000,
      )
      output_file.write(response.choices[0].message.content + '\n')
  output_file.close()
