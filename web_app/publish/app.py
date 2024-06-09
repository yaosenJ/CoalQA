# isort: skip_file
import os
import copy
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional
import shutil
import streamlit as st
import torch
from torch import nn
from transformers.generation.utils import (LogitsProcessorList,
                                           StoppingCriteriaList)
from transformers.utils import logging

from transformers import AutoTokenizer, AutoModelForCausalLM  # isort: skip

import sys
sys.path.append('/path/to/my_module/CoalQA') #注意这里/path/to/my_module换成你下载CoalQA仓库的父目录路径

from rag.main import setup_model_and_tokenizer
from rag.pipeline import CoalLLMRAG

logger = logging.get_logger(__name__)


@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    max_length: int = 32768
    top_p: float = 0.8
    temperature: float = 0.8
    do_sample: bool = True
    repetition_penalty: float = 1.005


@torch.inference_mode()
def generate_interactive(
    model,
    tokenizer,
    prompt,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor],
                                                List[int]]] = None,
    additional_eos_token_id: Optional[int] = None,
    **kwargs,
):
    inputs = tokenizer([prompt], padding=True, return_tensors='pt')
    input_length = len(inputs['input_ids'][0])
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    input_ids = inputs['input_ids']
    _, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
    if generation_config is None:
        generation_config = model.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    bos_token_id, eos_token_id = (  # noqa: F841  # pylint: disable=W0612
        generation_config.bos_token_id,
        generation_config.eos_token_id,
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)
    has_default_max_length = kwargs.get(
        'max_length') is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using 'max_length''s default ({repr(generation_config.max_length)}) \
                to control the generation length. "
            'This behaviour is deprecated and will be removed from the \
                config in v5 of Transformers -- we'
            ' recommend using `max_new_tokens` to control the maximum \
                length of the generation.',
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + \
            input_ids_seq_length
        if not has_default_max_length:
            logger.warn(  # pylint: disable=W4902
                f"Both 'max_new_tokens' (={generation_config.max_new_tokens}) "
                f"and 'max_length'(={generation_config.max_length}) seem to "
                "have been set. 'max_new_tokens' will take precedence. "
                'Please refer to the documentation for more information. '
                '(https://huggingface.co/docs/transformers/main/'
                'en/main_classes/text_generation)',
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = 'input_ids'
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, "
            f"but 'max_length' is set to {generation_config.max_length}. "
            'This can lead to unexpected behavior. You should consider'
            " increasing 'max_new_tokens'.")

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None \
        else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None \
        else StoppingCriteriaList()

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=stopping_criteria)
    logits_warper = model._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False)
        unfinished_sequences = unfinished_sequences.mul(
            (min(next_tokens != i for i in eos_token_id)).long())

        output_token_ids = input_ids[0].cpu().tolist()
        output_token_ids = output_token_ids[input_length:]
        for each_eos_token_id in eos_token_id:
            if output_token_ids[-1] == each_eos_token_id:
                output_token_ids = output_token_ids[:-1]
        response = tokenizer.decode(output_token_ids)

        yield response
        # stop when each sentence is finished
        # or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(
                input_ids, scores):
            break


def on_btn_click():
    del st.session_state.messages


@st.cache_resource
def load_model(generation_config):
    base_path = './CoalMineLLM_InternLM2'
    if not os.path.exists(base_path):
        os.system('apt install git')
        os.system('apt install git-lfs')
        #os.system(f'git clone https://code.openxlab.org.cn/viper/CoalMineLLM_InternLM2-Chat-1_8B.git {base_path}')
        os.system(f'git clone https://code.openxlab.org.cn/viper/CoalMineLLM_InternLM2-Chat-1_8B.git {base_path}')
        #os.system(f'git clone https://code.openxlab.org.cn/milowang/CoalMineLLM_InternLM2-Chat-7B-4bit.git {base_path}')
        os.system(f'cd {base_path} && git lfs pull')

    model, tokenizer, llm = setup_model_and_tokenizer(base_path, generation_config)
    return model, tokenizer, llm
  
@st.cache_resource
def load_multi_query_model(): 
    base_path = r'./CoalMineLLM_InternLM2'
    model = (AutoModelForCausalLM.from_pretrained(base_path,
                                                  trust_remote_code=True).to(
                                                      torch.bfloat16).cuda())
    tokenizer = AutoTokenizer.from_pretrained(base_path,
                                              trust_remote_code=True)
    return model, tokenizer

def get_similar_query(chat_model, chat_tokenizer, query, num=1):
    results = []
    for _ in range(0, num):
        # 大模型进行改写，记得do_sample设置成true，不然会输出同一个句子，缺少多样性
        response, _ = chat_model.chat(chat_tokenizer, query + "。你是一个改写句子的专家，注意：现在你的任务是改写/重写句子！！！！！，所以请你用另一种表达方式改写上述话。", history=[], do_sample=True, num_beams=3,
                                      temperature=0.8)
        results.append(response)
    return results



def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider('Max Length',
                               min_value=8,
                               max_value=32768,
                               value=2048)
        top_p = st.slider('Top P', 0.0, 1.0, 0.75, step=0.01)
        temperature = st.slider('Temperature', 0.0, 1.0, 0.1, step=0.01)
        st.button('Clear Chat History', on_click=on_btn_click)

    generation_config = GenerationConfig(max_length=max_length,
                                         top_p=top_p,
                                         temperature=temperature)

    return generation_config


user_prompt = '<|im_start|>user\n{user}<|im_end|>\n'
robot_prompt = '<|im_start|>assistant\n{robot}<|im_end|>\n'
cur_query_prompt = '<|im_start|>user\n{user}<|im_end|>\n\
    <|im_start|>assistant\n'


def combine_history(prompt, retrieval_content=''):
    messages = st.session_state.messages
    prompt = f"你需要根据以下检索到的专业知识:`{retrieval_content}`。从一个煤矿安全专家的专业角度来回答后续提问：{prompt}"
    meta_instruction = ('你是A100换你AD钙奶团队研发的煤矿安全领域大语言模型。旨在为煤矿企业负责人、安全管理人员、矿工等用户提供关于煤矿事故、煤矿安全规程规章制度及相关安全知识的智能问答服务。')
    total_prompt = f"<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"
    for message in messages:
        cur_content = message['content']
        if message['role'] == 'user':
            cur_prompt = user_prompt.format(user=cur_content)
        elif message['role'] == 'robot':
            cur_prompt = robot_prompt.format(robot=cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
    return total_prompt

def use_rag(rag_obj, prompt):
    # TODO: RAG function
    prompts = [prompt]
    retrieval_content = rag_obj.get_retrieval_content(prompts)
    return retrieval_content

def main():
    generation_config = prepare_generation_config()
    print('load model begin.')
    model, tokenizer, llm = load_model(generation_config)
    print('load model end.')
    rag_obj = CoalLLMRAG(llm, retrieval_num=3, rerank_flag=False, select_num=3)
    # print('load rag_obj.')

    # st.title('💬 coal QA')

    with st.sidebar:
        is_arg = st.radio(
            "Whether use RAG for generate",
            ("Yes", "No")
        )
        st.image(r"images/coal_mine_safety.png")
      
    robot_avator = "images/robot.jpg"
    st.title('💬 煤矿安全大模型--矿途智护者')
    

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role'], avatar=message.get('avatar')):
            st.markdown(message['content'])

    # Accept user input
    if prompt := st.chat_input('What is up?'):
        # Display user message in chat message container

        # print('multi_query')
        # query_model, query_tokenizer = load_multi_query_model()
        # query_model.cuda()
        # similar_prompts = get_similar_query(query_model, query_tokenizer, prompt)
        # query_model.cpu()
        # prompts = [prompt] + similar_prompts
        
        with st.chat_message('user'):
            st.markdown(prompt)

        

        if is_arg=="Yes":
            retrieval_content = use_rag(rag_obj, prompt)
        
            real_prompt = combine_history(prompt, retrieval_content)
        else:
            real_prompt = combine_history(prompt)
        # Add user message to chat history
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt,
        })

        with st.chat_message('robot',avatar=robot_avator):
            message_placeholder = st.empty()
            for cur_response in generate_interactive(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=real_prompt,
                    additional_eos_token_id=92542,
                    **asdict(generation_config),
            ):
                # Display robot response in chat message container
                message_placeholder.markdown(cur_response + '▌')
            message_placeholder.markdown(cur_response)
        # Add robot response to chat history
        st.session_state.messages.append({
            'role': 'robot',
            'content': cur_response,  # pylint: disable=undefined-loop-variable
            "avatar": robot_avator,
        })
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
