"""
setting.py 配置管理模块
利用Pydantic Settings 实现类型安全的配置
Literal - 限制变量只能取指定的几个字面量值，做字面量类型校验
"""

from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    """
    配置管理类
    """
    # 项目基本配置
    PROJECT_NAME:str = "企业级RAG智能问答系统"
    PROJECT_VERSION:str = "1.0.0"
    BASE_DIR: Path = Path(__file__).parent.parent.parent.parent.resolve()

    # 数据目录
    DATA_DIR : Path = BASE_DIR / "data"
    VECTOR_DB_DIR: Path = BASE_DIR / "vector_db"
    MODELS_DIR: Path = BASE_DIR / "models"
    UPLOAD_DIR: Path = BASE_DIR / "upload"

    # 向量数据库配置
    VECTOR_STORE_TYPE:Literal["faiss","chroma"] = "chroma"
    EMBEDDING_MODEL_NAME:str = "BAAI/bge-small-zh"
    EMBEDDING_DEVICE:str = "cpu"
    CHUNK_SIZE:int = 500
    CHUNK_OVERLAP:int = 50


    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = BASE_DIR / "app.log"

    # 安全配置
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8501"]
    SECRET_KEY: str = "your-secret-key"

    # 服务配置
    API_HOST: str = "localhost"
    API_PORT: int = 8000
    STREAMLIT_PORT: int = 8501


#实例化全局配置
settings = Settings()

# 确保必要目录存在
for dir_path in [settings.DATA_DIR,settings.VECTOR_DB_DIR,settings.MODELS_DIR,settings.UPLOAD_DIR]:
    dir_path.mkdir(parents=True,exist_ok=True)