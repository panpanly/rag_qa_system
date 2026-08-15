"""
检索模块 = 支持相似度检索 + 可以重排序
"""

import logging
import os
# 最前面！导入huggingface_hub之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download
from langchain_core.documents.compressor import BaseDocumentCompressor
from src.rag_qa_system.tools.vector_store import get_vector_store_manager
from src.rag_qa_system.config.setting import settings
from langchain.retrievers import ContextualCompressionRetriever


logger = logging.getLogger(__name__)

class CrossEncoderReranker(BaseDocumentCompressor):
    """ 基于交叉编码器的重排序器 """
    model_name:str
    top_k:int
    model:object = None # 延迟加载模型

    def __initialize_model(self):
        """初始化交叉编码器模型，使用时才第一加载"""
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(self.model_name)
                logger.info(f"交叉编码器模型已初始化：{self.model_name}")
            except Exception as e:
                logger.info(f"初始化交叉编码器模型失效：{str(e)}")
                self.model = None

class RagRetriever:
    """RAG检索器，封装向量检索和重排序功能"""
    def __init__(self):
        self.vector_store = get_vector_store_manager()
        self._download_model()
        self.reranker = self._initialize_reranker()

    def _download_model(self):
        """先下载模型到本地"""
        logger.info(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}")
        model_name = settings.RERANKER_MODEL_NAME
        local_model_path = settings.MODELS_DIR / model_name.replace("/", "_")

        if local_model_path.exists():
            logger.info(f"Reranker本地模型已存在: {local_model_path}")
            return

        logger.info(f"开始下载reranker模型 {model_name} 到 {local_model_path}")
        # 传入repo_id，指定保存目录
        snapshot_download(
            repo_id=model_name,
            local_dir=local_model_path,
            local_dir_use_symlinks=False,  # Windows必须False，不要软链接，下载真实文件
            resume_download=True,  # 断点续传
            force_download=False,
        )

    def _initialize_reranker(self):
        """初始化Reranker"""
        if not settings.USE_RERANKER:
            logger.info("未启用Reranker")
            return None
        try:
            # 尝试本地下载，本地没有再huggingface下载
            model_name = settings.RERANKER_MODEL_NAME
            local_model_path = settings.MODELS_DIR / model_name.replace("/","_")
            if local_model_path.exists():
                model_path = str(local_model_path)
                logger.info(f"从本地加载Reranker模型：{local_model_path}")
                reranker = CrossEncoderReranker(
                    model_name=model_path,
                    local_files_only=True,
                    top_k=settings.SEARCH_TOP_K
                )
            else:
                model_path = model_name
                logger.info(f"从HuggingFace下载Reranker模型：{model_name}")
                reranker = CrossEncoderReranker(
                    model_name=model_path,
                    top_k=settings.SEARCH_TOP_K
                )
            logger.info(f"Reranker已初始化：{settings.RERANKER_MODEL_NAME}")
            return reranker
        except Exception as e:
            logger.error(f"初始化Reranker失败: {str(e)}",exc_info=True)
            return None

    def get_compresstion_retriever(self, search_kwargs:dict):
        """
        获取适配的langchain链的检索器
        :param search_kwargs:
        :return:
        """
        search_kwargs = search_kwargs or {"k": settings.SEARCH_TOP_K}
        # 获取基础向量检索器
        base_retriever = self.vector_store._store.as_retriever(
            search_kwargs=search_kwargs
        )
        # 如果启动了reranker,则创建压缩检索器
        if self.reranker:
            return ContextualCompressionRetriever(
                base_compressor=self.reranker,
                base_retriever=base_retriever
            )
        else:
            return base_retriever


_rag_retriever_instance = None

def  get_rag_retriever() -> RagRetriever:
    global _rag_retriever_instance
    if _rag_retriever_instance is None:
        _rag_retriever_instance = RagRetriever()
    return _rag_retriever_instance