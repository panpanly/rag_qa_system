"""
FastApi 应用入口
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.rag_qa_system.config.setting import settings
from src.rag_qa_system.config.logging_config import setup_logging
from src.rag_qa_system.api.routes import upload_router,qa_router
from src.rag_qa_system.tools.vector_store import get_vector_store_manager
import uvicorn

setup_logging()

# 定义接口的生命周期函数
@asynccontextmanager
async def lifespan(app:FastAPI):
    """ 应用生命周期管理 """
    # 启动时初始化
    print(f"======服务启动，初始化向量数据库")
    get_vector_store_manager()
    yield
    print(f"===>>>>服务关闭，释放资源")




app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置CORS跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(upload_router,prefix="/api/v1")
app.include_router(qa_router,prefix="/api/v1")


# 接口安全校验
@app.get("/")
async def root():
    return {
        "messages":f"欢迎使用 {settings.PROJECT_NAME}",
        "version":settings.PROJECT_VERSION,
        "docs":"/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "rag_qa_system.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT
    )