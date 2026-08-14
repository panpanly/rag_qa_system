import os

import streamlit as st
import requests
import time
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title="知识库管理",
    page_icon="📁"
)

st.title("📁 知识库管理")

tab1,tab2,tab3 = st.tabs(["📤 上传文档","📊 知识库统计","⚡ 批量操作"])

with tab1:
    st.subheader("上传文档")

    uploaded_files = st.file_uploader(
        "选择文件（支持PDF、Word、Excel、TXT、Markdown）",
        type=["pdf", "docx", "doc", "txt", "md", "xlsx", "xls"],
        accept_multiple_files=True
    )

    category = st.selectbox(
        "文档分类",
        ["general", "技术文档", "管理制度", "产品手册", "培训材料", "其他"]
    )

    if st.button("🚀 开始上传并处理",type="primary",use_container_width=True):
        if not uploaded_files:
            st.warning("请选择文件")
        else:
            progress_bar = st.progress(0,text="准备上传...")
            status_text = st.empty()

            success_count = 0
            fail_count = 0

            for i, file in enumerate(uploaded_files):
                status_text.text(f"正在上传：{file.name} ({i+1}/{len(uploaded_files)})")
                progress_bar.progress((i) / len(uploaded_files))

            try:
                files = {"file":(file.name,file.getvalue())}
                data = {"category":category}
                response = requests.post(
                    f"{os.getenv('API_BASE')}/upload/file",
                    files=files,
                    data=data,
                    timeout=120
                )
                if response.status_code == 200:
                    success_count += 1
                    st.toast(f"✅ {file.name} 上传成功")
                else:
                    fail_count += 1
                    st.toast(f"❌ {file.name} 上传失败")
            except Exception as e:
                fail_count += 1
                st.toast(f"❌ {file.name} 上传失败")

            time.sleep(0.5)

            progress_bar.progress(1.0,text="上传完成!")
            status_text.text(f"上传完成！成功: {success_count}, 失败: {fail_count}")

            if success_count > 0:
                st.success(f"✅ 成功上传 {success_count} 个文件，正在后台处理...")







