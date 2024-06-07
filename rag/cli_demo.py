import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
from pipeline import GenerationConfig

model_name_or_path = "/group_share/internlm2_chat_7b_qlora_4000"  # 对于二代模型改为 zhangxiaobai_shishen2_full

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                             device_map='auto')
model = model.eval()

messages = []
generation_config = GenerationConfig(max_length=204, top_p=0.8, temperature=0.8, repetition_penalty=1.002)

response, history = model.chat(tokenizer, "甘肃华星煤业有限公司井下3183准备工作面回风巷发生一起运输伤亡事故。这件事是什么时候发生的?", history=[])
print(response)
response, history = model.chat(tokenizer, "请问您能提供有关山西兰花集团营山煤矿有限公司事故的信息吗?", history=history)
print(response)