"""
FastAPI依赖注入模块
"""
from src.rag_qa_system.tools.vector_store import get_vector_store_manager
from src.rag_qa_system.tools.rag_chain import get_rag_chain

def get_vector_store_dep():
    """获取向量库依赖"""
    return get_vector_store_manager()

def get_rag_chain_dep():
    """获取大模型链依赖"""
    return get_rag_chain()