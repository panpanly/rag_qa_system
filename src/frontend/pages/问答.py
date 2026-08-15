import os

import streamlit as st
import requests
import uuid
import json
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="智能问答",
    page_icon="💬"
)

st.title("💬 智能问答")
st.caption("基于知识库的智能问答，支持多轮对话")

# 保存会话信息到LocalStorage
def save_session():
    """保存会话列表和当前会话到浏览器LocalStorage"""
    if "session" in st.session_state and "current_session" in st.session_state:
        components.html(
            f"""
            <script>
                localStorage.setItem('rag_qa_sessions', JSON.stringify({json.dumps(st.session_state.sessions)}));
                localStorage.setItem('rag_qa_current_session', '{st.session_state.current_session}');
            </script>
            """,
            height=0
        )

# 从url参数中加载会话数据
def init_sessions_from_url():
    """ 从 URL 参数中初始化会话 """
    url_params = st.query_params
    if 'sessions' in url_params and 'current_session' in url_params:
        try:
            session_str = url_params['session']
            current_session = url_params['current_session']
            st.session_state.sessions = json.loads(session_str)
            st.session_state.current_session = current_session
            # 清除 URL 参数
            st.query_params.clear()
            return True
        except:
            pass
    return False


# 初始化会话状态
if not init_sessions_from_url():
    if "sessions" not in st.session_state:
        st.session_state.current_session = str(uuid.uuid4())
        st.session_state.sessions = {
            st.session_state.current_session:'新会话'
        }
        save_session()
    if "current_session" not in st.session_state:
        st.session_state.current_session = list(st.session_state.sessions.keys())[0]

if "messages" not in st.session_state:
    st.session_state.messages = []


# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 查看引用来源"):
                for src in msg["sources"]:
                    st.caption(f"**{src['source']}**(页码：{src.get('page','-')})")
                    st.text(src["content"][:200] + "..." if len(src["content"]) > 200 else src.get("content",""))


if prompt:= st.chat_input("请输入您的问题..."):
    # 添加用户消息
    st.session_state.messages.append({
        "role":"user",
        "content":prompt
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用api
    with st.chat_message("assistant"):
        with st.spinner("🤔 思考中..."):
            try:
                response = requests.post(
                    f"{os.getenv('API_BASE')}/qa/ask",
                    json={"question":prompt,"session_id":st.session_state.current_session},
                    timeout=60
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources",[])

                    st.markdown(answer)
                    if sources:
                        with st.expander("📚 引用来源"):
                            for src in sources:
                                st.caption(f"**{src['source']}**")
                                st.text(src["content"][:300] + "...." if len(src.get("content",'')) > 300 else src.get("content",""))

                    # 保存助手消息
                    st.session_state.messages.append({
                            "role":"assistant",
                            "content":answer,
                            "sources":sources
                        })
                else:
                    st.error(f"API错误（{response.status_code}:{response.text}）")
            except requests.exceptions.ConnectionError:
                st.error("❌ 无法连接到API服务，请确保后端已启动")
            except Exception as e:
                st.error(f"请求失败：{str(e)}")