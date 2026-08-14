
"""
embeeding 模型加载模块-采用单列模式，支持本地缓存，与HuggingFace自动下载
"""
import logging
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download
from langchain_huggingface import HuggingFaceEmbeddings
from src.rag_qa_system.config.setting import settings


class EmbeddingModel:
    _instance = None

    # 静态方法，用于创建类的实例
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel,cls).__new__(cls)
            cls._instance._download_model()
            cls._instance._initialize()
        return cls._instance

    def _download_model(self):
        """下载模型到本地"""
        model_name = settings.EMBEDDING_MODEL_NAME
        local_model_path = settings.MODELS_DIR / model_name.replace("/", "_")

        snapshot_download(
            repo_id=model_name,
            local_dir=local_model_path,
            local_dir_use_symlinks=False,  # windows必须关闭软链接
            resume_download=True
        )

    def _initialize(self):
        """初始化嵌入模型"""
        model_name = settings.EMBEDDING_MODEL_NAME
        local_model_path = settings.MODELS_DIR / model_name.replace("/","_")
        if local_model_path.exists():
            logging.info(f"从本地加载嵌入模型: {local_model_path}")
            model_path = str(local_model_path)
        else:
            logging.info(f"从HuggingFace下载嵌入模型:{model_name}")
            model_path = model_name

        try:
            self.model = HuggingFaceEmbeddings(
                model_name = model_path,
                model_kwargs = {"device":settings.EMBEDDING_DEVICE},
                encode_kwargs = {
                    "normalize_embeddings":True,
                    "batch_size":32,
                }
            )
            logging.info(f"嵌入模型已初始化：{model_name}")
        except Exception as e:
            logging.error(f"初始化嵌入模型失败：{str(e)}")
            raise

    def get_model(self):
        """ 返回嵌入模型实例 """
        return self.model

def get_embedding_model():
    """ 获取嵌入模型实例 """
    return EmbeddingModel().get_model()