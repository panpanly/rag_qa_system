"""
问答API
"""

from fastapi import APIRouter,Depends,FastAPI,HTTPException
from pydantic import BaseModel
from src.rag_qa_system.tools.rag_chain import RAGChain
from src.rag_qa_system.api.dependencies import get_rag_chain_dep

router = APIRouter(prefix="/qa",tags=["智能问答"])

class QuestionRequest(BaseModel):
    question:str
    session_id:str = "default"


class AnswerResponse(BaseModel):
    answer:str
    sources:list
    intent:str

class SourceInfo(BaseModel):
    source:str
    page:int = None
    content:str


@router.post("/ask",response_model=AnswerResponse)
async def ask_question(
    request:QuestionRequest,
    rag_chain:RAGChain = Depends(get_rag_chain_dep)
):
    """ 提交问题并获取答案"""
    if not request.question.strip():
        raise HTTPException(400,"问题不能为空")

    result = rag_chain.ask(question=request.question,session_id=request.session_id)
    return AnswerResponse(
        answer=result["answer"],
        sources=[SourceInfo(**s) for s in result["sources"]],
        intent = result["intent"]
    )