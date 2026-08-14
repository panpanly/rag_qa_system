import logging

from src.rag_qa_system.tools.embedding import get_embedding_model
from src.rag_qa_system.config.setting import settings
from langchain_community.vectorstores import Chroma,FAISS
from langchain.schema import Document

logging = logging.getLogger(__name__)

class VectorStoreManager:
    """ 向量库管理器 """
    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.store = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """ 初始化向量库 """
        if settings.VECTOR_STORE_TYPE == "chroma":
            # 如果是chroma向量数据库
            self._store = Chroma(
                persist_directory=str(settings.VECTOR_DB_DIR),
                embedding_function=self.embedding_model
            )
            logging.info(f"Chroma向量库已初始化，存储路径：{settings.VECTOR_DB_DIR}")
        elif settings.VECTOR_STORE_TYPE == 'faiss':
            faiss_index_path = settings.VECTOR_DB_DIR / "index.faiss"
            if faiss_index_path.exists():
                self._store = FAISS.load_local(
                    str(settings.VECTOR_DB_DIR),
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                logging.info(f"FAISS向量库已加载，存储路径：{settings.VECTOR_DB_DIR}")
            else:
                self._store = FAISS.from_documents(
                    [Document(page_content="初始化向量库",metadata={"source","init"})],
                    self.embedding_model
                )
                self._save_faiss()
                logging.info(f"FAISS向量库已保存，存储路径：{settings.VECTOR_DB_DIR}")
        else:
            raise ValueError(f"不支持的向量库类型：{settings.VECTOR_STORE_TYPE}")

    def _save_faiss(self):
        """保存Faiss向量库"""
        if settings.VECTOR_STORE_TYPE == 'faiss':
            self._store.save_local(str(settings.VECTOR_DB_DIR))

    def add_documents(self,documents:list) ->int:
        """添加文档切片到向量数据库"""
        if not documents:
            return 0

        try:
            self._store.add_documents(documents)
            if settings.VECTOR_STORE_TYPE == 'faiss':
                self._save_faiss()
            logging.info(f"成功添加{len(documents)}个文档到向量库")
            return len(documents)
        except Exception as e:
            logging.error(f"添加文档到向量库失败：{e}")
            return 0


_vector_store_instance:VectorStoreManager = None

def get_vector_store_manager() -> VectorStoreManager:
    """ 获取向量库管理器单例 """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreManager()
    else:
        return _vector_store_instance

