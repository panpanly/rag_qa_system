
from modelscope import snapshot_download
from src.rag_qa_system.config.setting import settings
import os

def download_bge_reranker():
    # 模型仓库id
    model_name = settings.RERANKER_MODEL_NAME
    # 本地保存路径，可以自定义
    local_model_path = settings.MODELS_DIR / model_name.replace("/", "_")

    # 如果目录不存在自动创建
    os.makedirs(local_model_path, exist_ok=True)

    print(f"开始下载模型 {model_name} 到 {local_model_path}")

    # snapshot_download 完整参数
    local_path = snapshot_download(
        repo_id=model_name,
        local_dir=local_model_path,
    )

    print(f"✅模型下载完成，本地路径：{local_path}")
    return local_path


if __name__ == "__main__":
    download_bge_reranker()