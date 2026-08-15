"""
多轮对话记忆管理模块 - 支持会话隔离
"""
import logging
import uuid
from datetime import datetime,timedelta
from langchain.memory import ConversationBufferWindowMemory


logger = logging.getLogger(__name__)

class ConversationSession:
    """ 单个会话的记忆管理 """
    def __init__(self,session_id:str, window_size:int = 5):
        self.session_id = session_id
        self.create_at = datetime.now()
        self.last_active = datetime.now()
        self.memory = ConversationBufferWindowMemory(
            k = window_size,  # k参数是指窗口大小 即最多保留多少条对话记录
            memory_key = "chat_history",  # memory_key是指对话历史的键名
            return_messages=True,  # return_messages是指是否返回消息列表
            output_key="answer" # output_key是指输出的键名
        )
        self.metadata:dict = {}

    def get_memory(self):
        """ 获取当前会话的记忆 """
        return self.memory.chat_memory.messages

    def add_user_message(self,message:str):
        """跟新用户历史信息"""
        self.memory.chat_memory.add_user_message(message=message)
        self.last_active = datetime.now()

    def add_ai_message(self,message:str):
        """ 添加AI信息 """
        self.memory.chat_memory.add_ai_message(message=message)
        self.last_active= datetime.now()

class MemoryManager:
    """ 多会话记忆管理 """
    def __init__(self, window_size: int = 5, session_ttl_minutes: int = 60):
        self.sessions:dict = {}
        self.window_size = window_size
        self.session_ttl = timedelta(minutes=session_ttl_minutes)
        logger.info(f"记忆管理初始化，window_size={window_size},session_ttl={session_ttl_minutes} minutes")

    def create_session(self,session_id:str):
        """创建新会话"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        if session_id in self.sessions:
            logger.warning(f"会话{session_id}已存在，将返回现有对话")
            return session_id
        session = ConversationSession(session_id, self.window_size)
        self.sessions[session_id] = session
        logger.info(f"创建新会话：{session_id}")
        return session_id

    def get_session(self, session_id:str):
        """获取会话"""
        session = self.sessions.get(session_id)
        if session:
            session.last_active = datetime.now()
        return session

    def get_chat_memory(self,session_id:str):
        """ 获取对话历史记录 """
        session = self.get_session(session_id)
        if not session:
            return []
        return session.get_memory()

    def get_or_create_session(self,session_id:str):
        """获取或创建对话"""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        new_session_id = self.create_session(session_id)
        return self.sessions[new_session_id]

    def add_exchange(self,session_id:str,question:str,answer:str):
        """更新历史对话"""
        session = self.get_or_create_session(session_id)
        session.add_user_message(question)
        session.add_ai_message(answer)
        logger.info(f"添加对话记录：会话{session_id},用户：{question},AI：{answer}")



# 全局单例
_memory_manager_instance = None

def get_memory_manager():
    """ 获取管理器单例 """
    global _memory_manager_instance
    if _memory_manager_instance is None:
        _memory_manager_instance = MemoryManager()
    return _memory_manager_instance