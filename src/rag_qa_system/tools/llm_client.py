"""
大模型客户端 - 封装ollama OpenAI SiliconFlow 等大模型客户端
"""

import logging

from openai import base_url

from src.rag_qa_system.config.setting import settings
from langchain_openai import ChatOpenAI


logger = logging.getLogger(__name__)

class LLMClient:
    """ 大模型客户端 """
    def __init__(self):
        self.llm = None
        self._initialize_client()

    def _initialize_client(self):
        """ 初始化大模型客户端 """
        try:
            if settings.LLM_PROVIDER == 'ollama':
                self._llm = ChatOpenAI(
                    model=settings.OLLAMA_MODEL_NAME,
                    base_url=settings.OLLAMA_BASE_URL,
                    temperature=0.1,
                    verbose=True
                )
                logger.info(f"Ollama 客户端已初始化: {settings.OLLAMA_MODEL_NAME}")
            elif settings.LLM_PROVIDER == 'openai':
                self._llm = ChatOpenAI(
                    model=settings.OPENAI_MODEL_NAME,
                    api_key=settings.OPENAI_API_KEY,
                    base_url=settings.OPENAI_BASE_URL,
                    temperature=0.1,
                    verbose=True
                )
            else:
                self._llm = ChatOpenAI(
                    model=settings.SILICONFLOW_MODEL_NAME,
                    api_key=settings.SILICONFLOW_API_KEY,
                    base_url=settings.SILICONFLOW_BASE_URL,
                    temperature=0.1,
                    verbose=True
                )
                logger.info(f"{settings.SILICONFLOW_MODEL_NAME} 客户端已初始化")

        except Exception as e:
            logger.error(f"初始化大模型客户端失败:{e}")
            raise

    def get_llm(self):
        """ 获取大模型 """
        return self._llm

"""单例实例化大模型"""
_llm_client = None

def get_llm():
    """ 获取大模型实例 """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client.get_llm()
