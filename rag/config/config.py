import os

cur_dir = os.path.dirname(os.path.abspath(__file__))                # config
src_dir = os.path.dirname(cur_dir)                                  # src
base_dir = os.path.dirname(src_dir)                          # base

# model
model_dir = os.path.join(base_dir, 'model')                         # model
embedding_path = os.path.join(model_dir, 'embedding_model')         # embedding
embedding_model_name = 'BAAI/bge-small-zh-v1.5'
rerank_path = os.path.join(model_dir, 'rerank_model')  	        	  # embedding
rerank_model_name = 'BAAI/bge-reranker-large'

# data
data_dir = os.path.join(base_dir, 'data')                           # data
knowledge_json_path = os.path.join(data_dir, 'knowledge.json')      # json
knowledge_pkl_path = os.path.join(data_dir, 'knowledge.pkl')        # pkl
doc_dir = os.path.join(data_dir, 'final_data.pdf')   
qa_dir = os.path.join(data_dir, 'json')   
cloud_vector_db_dir = os.path.join(base_dir, 'coal')

# log
log_dir = os.path.join(base_dir, 'log')                             # log
log_path = os.path.join(log_dir, 'log.log')                         # file

# txt embedding 切分参数     
chunk_size=1000
chunk_overlap=300

# vector DB
vector_db_dir = os.path.join(cloud_vector_db_dir, 'vector_db')

# RAG related
# select num: 代表rerank 之后选取多少个 documents 进入 LLM
# retrieval num： 代表从 vector db 中检索多少 documents。（retrieval num 应该大于等于 select num）
select_num = 3
retrieval_num = 3

# prompt
prompt_template = """
    你是一个乐于助人的问答代理人。\n
    你的任务是分析并综合检索回来的信息，从而提供有意义且高效的答案。
	{content}
	问题：{query}
"""
