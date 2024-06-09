# from util.llm import get_glm
from loguru import logger
import torch
import gc
import ctypes
from transformers.generation.streamers import TextStreamer
import argparse
from langchain_core.prompts import PromptTemplate
from transformers import PreTrainedTokenizerFast, StoppingCriteriaList
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
from rag.pipeline import CoalLLMRAG
from rag.stop_criteria import StopWordStoppingCriteria
from rag.config.config import prompt_template 
from rag.CoalLLM import CoalLLM
TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')
'''
	1）构建完整的 RAG pipeline。输入为用户 query，输出为 answer
	2）调用 embedding 提供的接口对 query 向量化
	3）下载基于 FAISS 预构建的 vector DB ，并检索对应信息
	4）调用 rerank 接口重排序检索内容
	5）调用 prompt 接口获取 system prompt 和 prompt template
	6）拼接 prompt 并调用模型返回结果

'''
def parse_args():
    parser = argparse.ArgumentParser(description='Chat with a HF model')
    parser.add_argument(
        'model_name_or_path', default='/group_share/internlm2_chat_7b_qlora_4000', help='Hugging Face model name or path')
    parser.add_argument(
        '--multi_query', default=False, help='whether to enable multi_query')
    adapter_group = parser.add_mutually_exclusive_group()
    adapter_group.add_argument(
        '--adapter', default=None, help='adapter name or path')
    adapter_group.add_argument(
        '--llava', default=None, help='llava name or path')
    parser.add_argument(
        '--visual-encoder', default=None, help='visual encoder name or path')
    parser.add_argument(
        '--visual-select-layer', default=-2, help='visual select layer')
    parser.add_argument('--image', default=None, help='image')
    parser.add_argument(
        '--torch-dtype',
        default='fp16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
        'a specific `dtype`.')
    # parser.add_argument(
    #     '--prompt-template',
    #     choices=PROMPT_TEMPLATE.keys(),
    #     default=None,
    #     help='Specify a prompt template')
    system_group = parser.add_mutually_exclusive_group()
    system_group.add_argument(
        '--system', default=None, help='Specify the system text')
    # system_group.add_argument(
    #     '--system-template',
    #     choices=SYSTEM_TEMPLATE.keys(),
    #     default=None,
    #     help='Specify a system template')
    parser.add_argument(
        '--bits',
        type=int,
        choices=[4, 8, None],
        default=None,
        help='LLM bits')
    parser.add_argument(
        '--bot-name', type=str, default='BOT', help='Name for Bot')
    parser.add_argument(
        '--with-plugins',
        nargs='+',
        choices=['calculate', 'solve', 'search'],
        help='Specify plugins to use')
    parser.add_argument(
        '--no-streamer', action='store_true', help='Whether to with streamer')
    parser.add_argument(
        '--lagent', action='store_true', help='Whether to use lagent')
    parser.add_argument(
        '--stop-words', nargs='+', type=str, default=[], help='Stop words')
    parser.add_argument(
        '--offload-folder',
        default=None,
        help='The folder in which to offload the model weights (or where the '
        'model weights are already offloaded).')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=128,
        help='Maximum number of new tokens allowed in generated text')
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='The value used to modulate the next token probabilities.')
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='The number of highest probability vocabulary tokens to '
        'keep for top-k-filtering.')
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help='If set to float < 1, only the smallest set of most probable '
        'tokens with probabilities that add up to top_p or higher are '
        'kept for generation.')
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.0,
        help='The parameter for repetition penalty. 1.0 means no penalty.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')
    args = parser.parse_args()
    return args

# def get_stop_criteria(
#     tokenizer,
#     stop_words=[],
# ):
#     stop_criteria = StoppingCriteriaList()
#     for word in stop_words:
#         stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
#     return stop_criteria


def setup_model_and_tokenizer(model_name, generation_config):
    # 加载模型和tokenizer
    model = (
            AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            .to(torch.bfloat16)
            .cuda()
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = CoalLLM(model, tokenizer)
    # print("完成本地模型的加载")
    model.generation_config.max_length = generation_config.max_length
    model.generation_config.top_p = generation_config.top_p
    model.generation_config.temperature = generation_config.temperature
    model.generation_config.repetition_penalty = generation_config.repetition_penalty
    return model, tokenizer, llm

def get_similar_query(chat_model, chat_tokenizer, query, num=1):
    results = []
    for _ in range(0, num):
        # 大模型进行改写，记得do_sample设置成true，不然会输出同一个句子，缺少多样性
        response, _ = chat_model.chat(chat_tokenizer, query + "。你是一个改写句子的专家，注意：现在你的任务是改写/重写句子！！！！！，所以请你用另一种表达方式改写上述话。", history=[], do_sample=True, num_beams=3,
                                      temperature=0.8)
        results.append(response)
    return results

def load_query_model(): 
    base_path = r'/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b-sft'
    model = (AutoModelForCausalLM.from_pretrained(base_path,
                                                  trust_remote_code=True).to(
                                                      torch.bfloat16).cuda())
    tokenizer = AutoTokenizer.from_pretrained(base_path,
                                              trust_remote_code=True)
    return model, tokenizer

def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()
# def process_user_input(prompt, model, tokenizer, llm, generation_config):
#     """
#     处理用户输入，根据用户输入内容调用相应的模型生成回复。

#     Args:
#         prompt (str): 用户输入的内容。
#         model (str): 使用的模型名称。
#         tokenizer (object): 分词器对象。
#         llm: langchain包装的大模型
#         generation_config (dict): 生成配置参数。
#     """
    
#     # 检查用户输入是否包含特定关键词
#     # keywords = ["怎么做", "做法", "菜谱"]
#     # contains_keywords = any(keyword in prompt for keyword in keywords)
    
#     # # 如果不包含关键词，立即返回错误响应
#     # if not contains_keywords:
#     #     error_response = "对不起，我不能帮助你处理这个问题。"
#     #     print("Robot:", error_response)
#     #     return

#     # 根据模型类型处理响应
#     if enable_rag:
#         cur_response = generate_interactive_rag(llm=llm, question=prompt, verbose=verbose)
#     else:
#         additional_eos_token_id = 103028 if base_model_type == 'internlm-chat-7b' else 92542
#         cur_response = generate_interactive(model=model, tokenizer=tokenizer, prompt=prompt,
#                                             additional_eos_token_id=additional_eos_token_id,
#                                             **generation_config)
#         cur_response = next(cur_response).replace('\\n', '\n')

#     # 输出机器人的回答
#     print("Robot:", cur_response)

if __name__ == "__main__":
    args = parse_args()

    def get_input():
        """Helper function for getting input from users."""
        sentinel = ''  # ends when this string is seen
        result = None
        while result is None:
            print(('\n请输入(双击结束输入) >>> '),
                end='')
            try:
                result = '\n'.join(iter(input, sentinel))
            except UnicodeDecodeError:
                print('Invalid characters detected. Please enter again.')
        return result

    generation_config = GenerationConfig(max_length=128)

    model, tokenizer, llm = setup_model_and_tokenizer(args.model_name_or_path, generation_config)
    del model, tokenizer
    clean_memory()
    while True:
        query = get_input()

        if args.multi_query:
            print('multi_query')
            query_model, query_tokenizer = load_query_model()
            query_model.cuda()
            similar_prompts = get_similar_query(query_model, query_tokenizer, prompt)
            query_model.cpu()
            prompts = [prompt] + similar_prompts

        else:
            prompts = [prompt]
        rag_obj = CoalLLMRAG(llm, retrieval_num=3, rerank_flag=False, select_num=3)


        res = rag_obj.main([query])

        logger.info(res)





