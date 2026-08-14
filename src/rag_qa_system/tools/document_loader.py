import logging
from pathlib import Path
from langchain_community.document_loaders import (
PyMuPDFLoader,
Docx2txtLoader,
TextLoader,
UnstructuredExcelLoader,
UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.rag_qa_system.config.setting import settings

logging = logging.getLogger(__name__)

class DocumentLoader:
    """统一文档加载器类"""
    LOADER_MAP = {
        ".pdf": PyMuPDFLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".xls": UnstructuredExcelLoader
    }

    def __init__(self,chunk_size:int = None,chunk_overlap:int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n","\n","。","！","？",".","!","?"," ",""],
            length_function=len
        )

    def load_file(self, file_path:str, metadata:dict = None) -> list:
        """加载单个文件"""
        file_path_obj = Path(file_path)
        if file_path_obj.exists():
            logging.error(f"文件不存在:{file_path_obj}")
            return []

        # 检查文件类型
        ext = file_path_obj.suffix.lower()
        if ext not in self.LOADER_MAP:
            logging.error(f"不支持的文件类型：{ext}")
            return []

        try:
           loader_class = self.LOADER_MAP[ext]
           if ext == '.txt':
               loader = loader_class(str(file_path_obj),encoding="utf-8")
           else:
               loader = loader_class(str(file_path_obj))
           documents = loader.load()
           logging.info(f"成功加载文件：{file_path_obj},原始文档书段数：{len(documents)}")

           # 添加自定义元数据
           base_metadata = {
               "source":str(file_path_obj),
               "file_name":file_path_obj.name,
               "file_type":ext[1:],
           }
           if metadata:
               base_metadata.update(metadata)
           for doc in documents:
               doc.metadata.update(base_metadata)

            # 分割文档
           split_docs = self.text_splitter.split_documents(documents)
           logging.info(f"成功分割文档：{file_path_obj},分割后的文档数：{len(split_docs)}")
           return split_docs

        except Exception as e:
            logging.error(f"加载文件失败：{file_path_obj},错误：{e}")
