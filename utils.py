from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders import PyMuPDFLoader
import streamlit as st
import tempfile, os

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)
        
def pdf_to_doc(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyMuPDFLoader(temp_filepath)
    docs = loader.load_and_split()
    return docs

def print_messages():
    if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
        for chat_messages in st.session_state["messages"]:
            st.chat_message(chat_messages.role).write(chat_messages.content)
