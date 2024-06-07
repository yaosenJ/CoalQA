import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

base_path = '../model/CoalMineLLM_InternLM2-Chat-1_8B'
os.system(f'git clone https://code.openxlab.org.cn/viper/CoalMineLLM_InternLM2-Chat-1_8B.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="CoalMineLLM_InternLM2-Chat-1_8B",
                description="""
                煤矿安全大模型————矿途智护者 
                 """,
                 ).queue(1).launch()
