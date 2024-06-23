import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

base_path = './model/internlm2-chat-7b'
# download repo to the base_path directory using git
os.system('apt install git')
os.system('apt install git-lfs')
os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-7b.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')
