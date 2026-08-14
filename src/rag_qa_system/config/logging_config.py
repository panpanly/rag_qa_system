"""
日志配置模块
日志级别 - DEBUG 、INFO、WARNING、ERROR
统一全局日志报错规则，主程序调用，其他任意模块继承全局日志
RotatingFileHandler - 按文件大小切割日志文件，当单个文件的日志文件到达设定的大小时，则自动新建下一个日志文件，好处的保留指定数量的历史日志，防止单个文件无限膨胀占满磁盘

"""

import logging
import sys
from src.rag_qa_system.config.setting import settings
from logging.handlers import RotatingFileHandler

def setup_logging():
    """ 配置全局日志函数 """
    log_format = "%(asctime)s -%(name)s -%(levelname)s -%(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging,settings.LOG_LEVEL))

    # 清空旧handler，解决重复打印，同时不受其他handler干扰
    root_logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging,settings.LOG_LEVEL))
    console_handler.setFormatter(logging.Formatter(log_format,datefmt=date_format))
    root_logger.addHandler(console_handler)

    # 文件处理器
    # 设置日志文件大小，当达到设定大小的时候，会自动轮转
    file_handler = RotatingFileHandler(
        settings.LOG_FILE,
        maxBytes=10*1024*1024, # 10MB
        backupCount=5, #保留5个备份
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format,datefmt=date_format))
    root_logger.addHandler(file_handler)

    # 减少第三方库的日志噪声
    logging.getLogger("https").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    # 测试日志
    root_logger.info(f"日志系统初始化完成, 级别是 {settings.LOG_LEVEL}")
