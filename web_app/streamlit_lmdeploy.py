# -*- coding: utf-8 -*-
import torch
import os
import streamlit as st
import json
import requests

def get_post(msg):
    post_data = json.dumps(msg)
    url = "http://127.0.0.1:23333/v1/chat/completions"
    response = requests.post(url, data=post_data)

    return response.text 

base_path = r'/group_share/internlm2_chat_7b_qlora_4000'

if not os.path.exists(base_path):
    pass
    # download repo to the base_path directory using git
    # os.system('apt install git')
    # os.system('apt install git-lfs')
    # os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b.git {base_path}')
    # os.system(f'cd {base_path} && git lfs pull')

os.system("lmdeploy serve api_server  %s --model-format hf  --quant-policy 0  --server-name 127.0.0.1 --server-port 23333 --tp 1" % base_path)

st.title("💬 Coal QA")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    #if not openai_api_key:
    #    st.info("Please add your OpenAI API key to continue.")
    #    st.stop()
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    messages=st.session_state.messages
    msg = get_post(messages)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
    
