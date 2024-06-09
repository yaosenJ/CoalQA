import json
import pickle
import os
import numpy as np

from loguru import logger
from langchain_community.vectorstores import FAISS
from rag.config.config import (
    embedding_path,
    embedding_model_name,
    doc_dir, qa_dir,
    knowledge_pkl_path,
    data_dir,
    vector_db_dir,
    rerank_path,
    rerank_model_name,
    chunk_size,
    chunk_overlap
)
from rag.pdf_read import FileOperation

from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from FlagEmbedding import FlagReranker

class Data_process():

    def __init__(self):

        self.chunk_size: int=chunk_size
        self.chunk_overlap: int=chunk_overlap
        
    def load_embedding_model(self, model_name=embedding_model_name, device='cpu', normalize_embeddings=True):
        """
        加载嵌入模型。
        
        参数:
        - model_name: 模型名称，字符串类型，默认为"BAAI/bge-small-zh-v1.5"。
        - device: 指定模型加载的设备，'cpu' 或 'cuda'，默认为'cpu'。
        - normalize_embeddings: 是否标准化嵌入向量，布尔类型，默认为 True。
        """
        if not os.path.exists(embedding_path):
            os.makedirs(embedding_path, exist_ok=True)
        # print(3)
        embedding_model_path = os.path.join(embedding_path,model_name.split('/')[1] + '.pkl')
        logger.info('Loading embedding model...')
        if os.path.exists(embedding_model_path):
            try:
                with open(embedding_model_path , 'rb') as f:
                    embeddings = pickle.load(f)
                    logger.info('Embedding model loaded.')
                    return embeddings
            except Exception as e:
                logger.error(f'Failed to load embedding model from {embedding_model_path}')
        try:
            # embeddings = HuggingFaceBgeEmbeddings(
            #     model_name=model_name,
            #     model_kwargs={'device': device},
            #     encode_kwargs={'normalize_embeddings': normalize_embeddings})
            # logger.info('Embedding model loaded.')
            # with open(embedding_model_path, 'wb') as file:
            #     pickle.dump(embeddings, file)
            os.system('apt install git')
            os.system('apt install git-lfs')
            #os.system(f'git clone https://code.openxlab.org.cn/viper/CoalMineLLM_InternLM2-Chat-1_8B.git {base_path}')
            os.system(f'git clone https://code.openxlab.org.cn/answer-qzd/bge_small_1.5.git {embedding_path}')
            #os.system(f'git clone https://code.openxlab.org.cn/milowang/CoalMineLLM_InternLM2-Chat-7B-4bit.git {base_path}')
            os.system(f'cd {embedding_path} && git lfs pull')
            os.system(f'rm -rf .gitattributes')

            with open(embedding_model_path , 'rb') as f:
                embeddings = pickle.load(f)
                logger.info('Embedding model loaded.')
        except Exception as e:
            logger.error(f'Failed to load embedding model: {e}')
            return None
        return embeddings
    
    def load_rerank_model(self, model_name=rerank_model_name):
        """
        加载重排名模型。
        
        参数:
        - model_name (str): 模型的名称。默认为 'BAAI/bge-reranker-large'。
        
        返回:
        - FlagReranker 实例。
        
        异常:
        - ValueError: 如果模型名称不在批准的模型列表中。
        - Exception: 如果模型加载过程中发生任何其他错误。
        
        """ 
        if not os.path.exists(rerank_path):
            os.makedirs(rerank_path, exist_ok=True)
        rerank_model_path = os.path.join(rerank_path, model_name.split('/')[1] + '.pkl')   
        logger.info('Loading rerank model...')
        if os.path.exists(rerank_model_path):
            try:
                with open(rerank_model_path , 'rb') as f:
                    reranker_model = pickle.load(f)
                    logger.info('Rerank model loaded.')
                    return reranker_model
            except Exception as e:
                logger.error(f'Failed to load embedding model from {rerank_model_path}') 
        try:
            os.system('apt install git')
            os.system('apt install git-lfs')
            #os.system(f'git clone https://code.openxlab.org.cn/viper/CoalMineLLM_InternLM2-Chat-1_8B.git {base_path}')
            os.system(f'git clone https://code.openxlab.org.cn/answer-qzd/bge_rerank.git {rerank_path}')
            #os.system(f'git clone https://code.openxlab.org.cn/milowang/CoalMineLLM_InternLM2-Chat-7B-4bit.git {base_path}')
            os.system(f'cd {rerank_path} && git lfs pull')

            with open(rerank_model_path , 'rb') as f:
                reranker_model = pickle.load(f)
                logger.info('Rerank model loaded.')
                
        except Exception as e:
            logger.error(f'Failed to load rerank model: {e}')
            raise

        return reranker_model
    
    def rerank(self, reranker, query, contexts, select_num):
        merge = [[query, context] for context in contexts]
        scores = reranker.compute_score(merge)
        sorted_indices = np.argsort(scores)[::-1]

        return [contexts[i] for i in sorted_indices[:select_num]]

    def extract_text_from_json(self, obj, content=None):
        """
        抽取json中的文本，用于向量库构建
        
        参数:
        - obj: dict,list,str
        - content: str
        
        返回:
        - content: str 
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                try:
                    content = self.extract_text_from_json(value, content)
                except Exception as e:
                    print(f"Error processing value: {e}")
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                try:
                    content = self.extract_text_from_json(item, content)
                except Exception as e:
                    print(f"Error processing item: {e}")
        elif isinstance(obj, str):
            content += obj
        return content

    def split_document(self, data_path):
        """
        切分data_path文件夹下面的所有pdf文件
        
        参数:
        - data_path: str
        
        返回：
        - split_docs: list
        """
        # text_spliter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        text_spliter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap) 
        split_docs = []
        logger.info(f'Loading pdf files from {data_path}')
        if os.path.isdir(data_path):
            loader = DirectoryLoader(data_path, glob="**/*.pdf",show_progress=True)
            docs = loader.load()
            split_docs = text_spliter.split_documents(docs)
        elif data_path.endswith('.pdf'): 
            file_path = data_path
            logger.info(f'splitting file {file_path}')
            file_opr = FileOperation()
            text, error = file_opr.read(file_path)
            if error is not None:
                logger.info(f'Error!!! {error}')
            docs = text_spliter.create_documents([text])
            splits = text_spliter.split_documents(docs)
            split_docs = splits
        
        elif data_path.endswith('.txt'): 
            file_path = data_path
            logger.info(f'splitting file {file_path}')
            # text_loader = TextLoader(file_path, encoding='utf-8')        
            # text = text_loader.load()
            with open(file_path,encoding='utf-8') as f:
                accidents_data_txt = f.read()
            docs = text_spliter.create_documents([accidents_data_txt])
            splits = text_spliter.split_documents(docs)
            split_docs = splits
        logger.info(f'split_docs size {len(split_docs)}')
        return split_docs
  
    def create_vector_db(self, emb_model):
        '''
        创建并保存向量库
        '''
        logger.info(f'Creating index...')
        split_doc = self.split_document(doc_dir)

        logger.info(f'FAISS.from_documents')
        db = FAISS.from_documents(split_doc, emb_model)
        logger.info(f'saving, {len(split_doc)}')
        db.save_local(vector_db_dir)
        return db
        
    def load_vector_db(self, knowledge_pkl_path=knowledge_pkl_path, doc_dir=doc_dir, qa_dir=qa_dir):
        '''
        加载向量库
        '''
     
        emb_model = self.load_embedding_model()
        if not os.path.exists(vector_db_dir) or not os.listdir(vector_db_dir):
            db = self.create_vector_db(emb_model)
        else:
            db = FAISS.load_local(vector_db_dir, emb_model, allow_dangerous_deserialization=True)
        return db
    
if __name__ == "__main__":
    logger.info(data_dir)
    if not os.path.exists(data_dir):
         os.mkdir(data_dir)   
    dp = Data_process()
    rerank_model = dp.load_rerank_model()

    db = dp.load_vector_db()

    query = "请问您能提供有关黑河市兴边矿业有限公司事故的信息吗?"
    documents = db.similarity_search(query, k=4)
    content = []     
    for doc in documents:
        content.append(doc.page_content)

    logger.info(f'Query: {query}')
    logger.info(f'Contexts length:{len(content)}')
    logger.info("Retrieve contexts:")
    logger.info(content)

    rerank_model = dp.load_rerank_model()
    documents = dp.rerank(rerank_model, query, content, 2)
    logger.info("After rerank...")
    reranked = []
    for doc in documents:
        reranked.append(doc)
    # logger.info("Retrieve contexts:")
    logger.info(reranked)
