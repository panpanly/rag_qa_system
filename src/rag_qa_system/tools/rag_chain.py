"""
RAG问答链模块 - 使用LCEL语法实现
整合、检索、历史感知、意图识别、LLM生成等模块，支持多轮对话和来源追溯
"""

import logging
from src.rag_qa_system.tools.llm_client import get_llm
from src.rag_qa_system.tools.intent_recognizer import IntentRecognizer
from src.rag_qa_system.tools.memory_manager import get_memory_manager
from src.rag_qa_system.tools.retriever import get_rag_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from src.rag_qa_system.config.setting import settings

logger = logging.getLogger(__name__)

DEFAULT_SESSION_ID = "default"

class RAGChain:
    """ rag问答链 """
    def __init__(self):
        # 初始化大模型
        self.llm = get_llm()
        #意图识别
        self.intent_recognizer = IntentRecognizer()
        # 历史记忆管理
        self.memory_manager = get_memory_manager()
        # 初始化重排序
        self.retriever = get_rag_retriever()
        # 创建模型链
        self._rag_chain = self._create_rag_chain()
        logger.info(f"RAG问答链初始化完成")

    def _create_rag_chain(self):
        """创建RAG问答链"""
        # 1、改写用户提示词
        contextualize_q_system_prompt = """
            你是一个专业的query改写专家，你的任务是根据历史对话，将用户的追问改写成一个独立，完整的query，
            如果用户的提问已经足够独立，请直接返回原问题，不要添加任何额外内容
        """

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system",contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human","{input}"),
        ])


        # 创造历史感知检测器
        history_aware_retriever = create_history_aware_retriever(
            self.llm,
            self.retriever.get_compresstion_retriever(search_kwargs={"k": settings.SEARCH_TOP_K}),
            contextualize_q_prompt,
        )

        logger.info(f"历史感知检索器创建完成")

        # 2、问答生成提示词
        qa_system_prompt = """
            你是一个专业的企业知识助手，请根据以下提供的上下文信息回答用户问题。
            【重要规则】
            - 对于与企业知识库相关的问题，优先使用上下文中的内容。
            - 对于不需要知识库的问题（如数学计算、常识性问题等），可以直接回答。
            - 如果上下文不足以回答与知识库相关的问题，请明确回复：“根据现有知识库无法回答该问题”。
            - 回答时请保持专业、客观，避免使用主观观点。
            - 不要输出任何乱码、特殊符号或不连贯的词语。
            【上下文信息】
            {context}
        """

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",qa_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human","{input}"),
            ]
        )

        # 创建文档合并链
        question_answer_chain = create_stuff_documents_chain(
            self.llm,
            qa_prompt
        )

        # 3.组装完整的RAG链
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )
        logger.info(f"RAG问答链创建完成")
        return rag_chain

    def ask(self,question:str, session_id:str = DEFAULT_SESSION_ID) -> dict:
        """
        处理用户问题
        :param question:
        :param sesstion_id:
        :return:
        """
        if not question.strip():
            return {"answer":"问题不能为空", "sources":[], "intent":"无效输入"}

        # 1、识别意图
        intent = self.intent_recognizer.recognize(question)
        logger.info(f"识别到的意图：{intent}")

        # 2、获取历史对话消息
        try:
            chat_history = self.memory_manager.get_chat_memory(session_id)
            chain_input = {
                "input":question,
                "chat_history":chat_history
            }
            result = self._rag_chain.invoke(chain_input)
            answer = result["answer"]
            source_docs = result.get("context",[])
            sources = []
            for doc in source_docs:
                sources.append({
                    "source":doc.metadata.get("file_name",doc.metadata.get("source","未知")),
                    "page":doc.metadata.get("page",1),
                    "content":doc.page_content[:200] + '...'  if len(doc.page_content) > 200 else doc.page_content
                })
            # 3、更新对话历史
            self.memory_manager.add_exchange(session_id,question,answer)
            return {
                "answer":answer,
                "sources":sources,
                "intent":intent
            }
        except Exception as e:
            logger.error(f"处理问题时出错：{str(e)}")
            return {
                "answer":"抱歉，处理问题时出错了",
                "sources":[],
                "intent":"系统错误"
            }



_rag_chain_instance = None

def get_rag_chain() -> RAGChain:
    """ 获取RAG问答链单例"""
    global _rag_chain_instance
    if _rag_chain_instance is None:
        _rag_chain_instance = RAGChain()
    return _rag_chain_instance