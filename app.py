
# # import os
# # import streamlit as st
# # from backend import create_pdf_retriever, generate_answer

# # # ==============================
# # # 📁 PDF Upload Override System
# # # ==============================
# # UPLOAD_DIR = "uploads"
# # os.makedirs(UPLOAD_DIR, exist_ok=True)


# # def override_pdf_upload(uploaded_file):
# #     """
# #     Replaces any previously uploaded PDFs with the new one.
# #     Deletes old files, saves the new one, and rebuilds the retriever.
# #     """
# #     # 🧹 Step 1: Clear all old PDFs
# #     for old_file in os.listdir(UPLOAD_DIR):
# #         old_path = os.path.join(UPLOAD_DIR, old_file)
# #         if os.path.isfile(old_path):
# #             try:
# #                 os.remove(old_path)
# #             except Exception as e:
# #                 st.warning(f"⚠️ Could not delete {old_file}: {e}")

# #     # 💾 Step 2: Save new uploaded file
# #     file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
# #     with open(file_path, "wb") as f:
# #         f.write(uploaded_file.read())

# #     # 🧠 Step 3: Rebuild retriever
# #     st.session_state.retriever = create_pdf_retriever([file_path])
# #     st.session_state.messages = []  # clear chat

# #     # ✅ Step 4: Success message
# #     st.success(f"✅ Uploaded and overridden: {uploaded_file.name}")


# # # ==============================
# # # 🌟 Streamlit UI
# # # ==============================
# # st.set_page_config(page_title="📚 Multi-PDF Chatbot", page_icon="🤖")

# # st.title("📚 Chat with Your PDF")
# # st.caption("Upload a PDF — each new upload will override the previous one.")

# # uploaded_file = st.file_uploader(
# #     "Upload your PDF file",
# #     type=["pdf"],
# #     accept_multiple_files=False,  
# #     key="pdf_uploader"
# # )

# # if "retriever" not in st.session_state:
# #     st.session_state.retriever = None
# # if "messages" not in st.session_state:
# #     st.session_state.messages = []

# # # 🏗️ Handle new uploads
# # if uploaded_file:
# #     override_pdf_upload(uploaded_file)

# # # 💬 Chat Section
# # if st.session_state.retriever:
# #     st.subheader(" Chat with your PDF")

# #     for msg in st.session_state.messages:
# #         with st.chat_message(msg["role"]):
# #             st.markdown(msg["content"])

# #     if user_input := st.chat_input("Ask a question about your PDF..."):
# #         st.session_state.messages.append({"role": "user", "content": user_input})
# #         with st.chat_message("user"):
# #             st.markdown(user_input)
# #         with st.chat_message("assistant"):
# #             with st.spinner("Searching your PDF..."):
# #                 answer = generate_answer(user_input, st.session_state.retriever)
# #                 st.markdown(answer)
# #         st.session_state.messages.append({"role": "assistant", "content": answer})
# # else:
# #     st.info("Please upload a PDF to start chatting.")



# # import os
# # import streamlit as st
# # from backend import create_pdf_retriever, generate_answer

# # # Settings for appearance and session
# # st.set_page_config(page_title="HR RAG Chatbot", page_icon=":robot_face:", layout="wide")

# # # --- Sidebar/UI Controls ---
# # with st.sidebar:
# #     st.title("Settings")
# #     api_url = st.text_input("API Base URL", value="http://localhost:8000")
# #     top_k = st.slider("Top-K Chunks", 1, 10, 5)
# #     attach_context = st.checkbox("Attach Context Under Answer", value=False)
# #     attach_sources = st.checkbox("Attach Sources Under Answer", value=True)
# #     st.write("")
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         if st.button("Check API Health"):
# #             try:
# #                 import requests
# #                 resp = requests.get(f"{api_url}/")
# #                 st.success(resp.json().get("message", "API healthy!"))
# #             except Exception as e:
# #                 st.error(f"Failed: {e}")
# #     with col2:
# #         if st.button("Clear Chat"):
# #             st.session_state.messages = []

# #     st.write("---")
# #     uploaded_file = st.file_uploader(
# #         "Attach HR Document (PDF)", type=["pdf"], accept_multiple_files=False, key="pdf_uploader"
# #     )

# # # --- Session state ---
# # if "retriever" not in st.session_state:
# #     st.session_state.retriever = None
# # if "messages" not in st.session_state:
# #     st.session_state.messages = []

# # # --- PDF Upload Handling (override mode) ---
# # UPLOAD_DIR = "uploads"
# # os.makedirs(UPLOAD_DIR, exist_ok=True)

# # def override_pdf_upload(uploaded_file):
# #     for old_file in os.listdir(UPLOAD_DIR):
# #         old_path = os.path.join(UPLOAD_DIR, old_file)
# #         if os.path.isfile(old_path):
# #             try:
# #                 os.remove(old_path)
# #             except Exception as e:
# #                 st.warning(f"⚠️ Could not delete {old_file}: {e}")
# #     file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
# #     with open(file_path, "wb") as f:
# #         f.write(uploaded_file.read())
# #     st.session_state.retriever = create_pdf_retriever([file_path])
# #     st.session_state.messages = []
# #     st.success(f"✅ Uploaded and overridden: {uploaded_file.name}")

# # if uploaded_file:
# #     override_pdf_upload(uploaded_file)

# # # --- Main UI ---
# # st.markdown(
# #     """
# #     <div style='text-align:center'>
# #         <h1>🤖 HR RAG Chatbot</h1>
# #         <p>Chat with your HR knowledge base.</p>
# #         <div style='background-color: #ffe6a1; display:inline-block; padding:12px 24px; border-radius:8px; margin-bottom:12px;'>
# #             <strong>Hi! I'm your HR RAG assistant. Ask me about your HR documents.</strong>
# #         </div>
# #     </div>
# #     """,
# #     unsafe_allow_html=True,
# # )

# # # --- Chat area ---
# # st.write("")

# # chat_container = st.container()
# # if st.session_state.retriever:
# #     with chat_container:
# #         st.subheader("Chat mode")
# #         # Display chat bubbles
# #         for msg in st.session_state.messages:
# #             with st.chat_message(msg["role"]):
# #                 st.markdown(msg["content"])
# #         # User input at the bottom
# #         if user_input := st.chat_input("Type your question..."):
# #             st.session_state.messages.append({"role": "user", "content": user_input})
# #             with st.chat_message("user"):
# #                 st.markdown(user_input)
# #             with st.chat_message("assistant"):
# #                 with st.spinner("Searching your knowledge base..."):
# #                     # Use Top-K value
# #                     retriever = st.session_state.retriever
# #                     retriever.search_kwargs["k"] = top_k
# #                     answer = generate_answer(user_input, retriever)
# #                     st.markdown(answer)
# #             st.session_state.messages.append({"role": "assistant", "content": answer})
# # else:
# #     st.info("Please upload a PDF to start chatting.")





# import os
# import streamlit as st
# from backend import create_pdf_retriever, generate_answer

# # ========== Streamlit UI Layout ==========
# st.set_page_config(page_title="HR RAG Chatbot", page_icon=":robot_face:", layout="wide")

# # --- Sidebar ---
# with st.sidebar:
#     st.title("Settings")
#     api_url = st.text_input("API Base URL", value="http://localhost:8000")
#     top_k = st.slider("Top-K Chunks", 1, 10, 5)
#     attach_context = st.checkbox("Attach Context Under Answer", value=False)
#     attach_sources = st.checkbox("Attach Sources Under Answer", value=True)
#     st.write("")
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("Check API Health"):
#             try:
#                 import requests
#                 resp = requests.get(f"{api_url}/")
#                 st.success(resp.json().get("message", "API healthy!"))
#             except Exception as e:
#                 st.error(f"Failed: {e}")
#     with col2:
#         if st.button("Clear Chat"):
#             st.session_state.messages = []
#     st.write("---")
#     uploaded_file = st.file_uploader(
#         "Attach HR Document (PDF)", type=["pdf"], accept_multiple_files=False, key="pdf_uploader"
#     )

# # --- Session State ---
# if "retriever" not in st.session_state:
#     st.session_state.retriever = None
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# def override_pdf_upload(uploaded_file):
#     for old_file in os.listdir(UPLOAD_DIR):
#         old_path = os.path.join(UPLOAD_DIR, old_file)
#         if os.path.isfile(old_path):
#             try:
#                 os.remove(old_path)
#             except Exception as e:
#                 st.warning(f"⚠️ Could not delete {old_file}: {e}")
#     file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.read())
#     st.session_state.retriever = create_pdf_retriever([file_path])
#     st.session_state.messages = []
#     st.success(f"✅ Uploaded and overridden: {uploaded_file.name}")

# if uploaded_file:
#     override_pdf_upload(uploaded_file)

# # --- Header section ---
# st.markdown(
#     """
#     <div style='text-align:center'>
#         <h1>🤖 HR RAG Chatbot</h1>
#         <p>Chat with your HR knowledge base.</p>
#         <div style='background-color: #ffe6a1; display:inline-block; padding:12px 24px; border-radius:8px; margin-bottom:12px;'>
#             <strong>Hi! How can I help you?</strong>
#         </div>
#     </div>
#     """, unsafe_allow_html=True
# )

# # --- Display Chat Conversation (Chronological, bottom input) ---
# container = st.container()
# with container:
#     # Display chat history, user/assistant separated
#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"], unsafe_allow_html=True)

# # --- Chat Input at Bottom ---
# user_input = st.chat_input("Type your question...")  # This is always at the bottom!
# if user_input and st.session_state.retriever:
#     st.session_state.messages.append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.markdown(user_input)
#     with st.chat_message("assistant"):
#         with st.spinner("Searching..."):
#             # Update top_k for retriever
#             st.session_state.retriever.search_kwargs["k"] = top_k
#             answer = generate_answer(
#                 user_input, 
#                 st.session_state.retriever,
#                 include_sources=attach_sources
#             )
#             st.markdown(answer, unsafe_allow_html=True)
#     st.session_state.messages.append({"role": "assistant", "content": answer})

# if not st.session_state.retriever:
#     st.info("Please upload a PDF to start chatting.")

