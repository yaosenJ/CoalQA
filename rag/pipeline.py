from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from transformers.utils import logging

from rag.data_generate import Data_process
from rag.config.config import prompt_template 

from dataclasses import dataclass
from typing import Callable, List, Optional
import torch
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.output_parsers import BooleanOutputParser
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain, RetrievalQA

logger = logging.get_logger(__name__)


class CoalLLMRAG(object):
    """
        CoalLLM RAG Pipeline
            1. 根据 query 进行 embedding
            2. 从 vector DB 中检索数据
            3. rerank 检索后的结果
            4. 将 query 和检索回来的 content 传入 LLM 中
    """

    def __init__(self, model, retrieval_num=3, rerank_flag=False, select_num=1) -> None:
        """
            输入 Model 进行初始化 

            DataProcessing obj: 进行数据处理，包括数据 embedding/rerank
            vectorstores: 加载vector DB。如果没有应该重新创建
            system prompt: 获取预定义的 system prompt
            prompt template: 定义最后的输入到 LLM 中的 template

        """
        self.model = model
        self.data_processing_obj = Data_process()
        self.vectorstores = self._load_vector_db()
        self.prompt_template = prompt_template
        self.retrieval_num = retrieval_num
        self.rerank_flag = rerank_flag
        self.select_num = select_num

    def _load_vector_db(self):
        """
            调用 embedding 模块给出接口 load vector DB
        """
        vectorstores = self.data_processing_obj.load_vector_db()

        return vectorstores 

    def get_retrieval_content(self, querys) -> str:
        """
            Input: 用户提问, 是否需要rerank
            ouput: 检索后的内容        
        """
        # print(querys)
        output = []
        content = []
        for query in querys:
            
            documents = self.vectorstores.similarity_search(query, k=self.retrieval_num)
            
            for doc in documents:
                content.append(doc.page_content)
            logger.info(f'Contexts length:{len(content)}')
            if self.rerank_flag:
                model = self.data_processing_obj.load_rerank_model()
                documents = self.data_processing_obj.rerank(model, query, content, self.select_num)

                for doc in documents:
                    output.append(doc)
                logger.info(f'Selected contexts length:{len(output)}')
                logger.info(f'Selected contexts: {output}')
            else:
                logger.info(f'Selected contexts: {content}')
        return output if self.rerank_flag else content
    
    
    def generate_answer(self, query, content) -> str:
        """
            Input: 用户提问， 检索返回的内容
            Output: 模型生成结果
        """
        prompt_template = """
            你是一个乐于助人的问答代理人。\n
            你的任务是分析并综合检索回来的信息，从而提供有意义且高效的答案。
            {content}
            问题：{query}
        """
        # 构建 template 
        # 第一版不涉及 history 信息，因此将 system prompt 直接纳入到 template 之中
        prompt = PromptTemplate(
                template=self.prompt_template,
                input_variables=["query", "content"],
                )

        # 定义 chain
        # output格式为 string
        rag_chain = prompt | self.model | StrOutputParser()

        # Run
        generation = rag_chain.invoke(
                {
                    "query": query,
                    "content": content,
                }
            )
        return generation
    
    def main(self, query) -> str:
        """
            Input: 用户提问
            output: LLM 生成的结果

            定义整个 RAG 的 pipeline 流程，调度各个模块
            TODO:
                加入 RAGAS 评分系统，精排模型以及向量训练
        """
        content = self.get_retrieval_content(query)
        response = self.generate_answer(query, content)

        return response
