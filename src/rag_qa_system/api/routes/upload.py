import uuid
from pathlib import Path
from fastapi import APIRouter,UploadFile,File,Form,HTTPException,BackgroundTasks
from fastapi import Depends
from fastapi.responses import JSONResponse
from src.rag_qa_system.tools.vector_store import VectorStoreManager
from src.rag_qa_system.api.dependencies import get_vector_store_dep
import logging
from src.rag_qa_system.config.setting import settings
import shutil
from src.rag_qa_system.tools.document_loader import DocumentLoader


logging = logging.getLogger(__name__)
router = APIRouter(prefix="/upload",tags=["文件上传"])

def process_file_background(file_path:str, metadata:dict, vector_store:VectorStoreManager):
    """ 后台处理文件并添加到向量库 """
    try:
        # 不同类型文件的解析
        loader = DocumentLoader()
        # 加载文件
        docs = loader.load_file(file_path,metadata)
        if docs:
            vector_store.add_documents(docs)
            logging.warning(f"后台处理完成：{file_path},添加{len(docs)}个片段")
        else:
            logging.warning(f"文件无有效内容：{file_path}")
    except Exception as e:
        logging.error(f"后台处理文件失败：{file_path},错误：{str(e)}")


@router.post("/file")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form("general"),
    vector_store: VectorStoreManager = Depends(get_vector_store_dep)
):
    """
    上传单个文件
    :param background_tasks:
    :param file:
    :param category:
    :param vector_store:
    :return:
    """
    # 验证文件类型
    allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md', '.xlsx', '.xls'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(400, f"不支持的文件格式:{file_ext}")

    # 生成唯一文件名
    unique_id = uuid.uuid4().hex[:8]
    safe_filename = f"{unique_id}_{file.filename}"
    save_path = settings.UPLOAD_DIR / safe_filename

    # 保存文件到本地
    try:
        with open(save_path,'wb') as buffer:
            shutil.copyfileobj(file.file,buffer)
    except Exception as e:
        logging.error(f"文件保存失败：{e}")
        raise HTTPException(500,f"文件保存失败：{str(e)}")


    # 元数据
    metadata = {
        "category":category,
        "original_filename":file.filename,
        "upload_time":str(Path(save_path).stat().st_ctime)
    }

    # 后台处理添加到向量数据库
    background_tasks.add_task(
        process_file_background,
        str(save_path),
        metadata,
        vector_store
    )

    return JSONResponse({
        "status":"success",
        "message":f"文件{file.filename}上传成功，正在后台处理",
        "file_id":unique_id,
        "filename":safe_filename
    })

