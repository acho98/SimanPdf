import streamlit as st
from utils import print_messages, pdf_to_doc, StreamHandler
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os, uuid

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

st.set_page_config(page_title="simanpdf", page_icon="")
st.title("심안(心眼)봇")

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state["store"]:
        st.session_state["store"][session_id] = ChatMessageHistory()
    return st.session_state["store"][session_id]

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "store" not in st.session_state:
    st.session_state["store"] = dict()

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()

session_id = st.session_state.id
client = None

with st.sidebar:
    st.header("Add your documents")

    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file is not None:
        uploaded_file_hash = hash(uploaded_file.getvalue())
        
        if 'uploaded_file_hash' not in st.session_state or st.session_state['uploaded_file_hash'] != uploaded_file_hash:
            with st.spinner("Processing the uploaded PDF..."):
                pages = pdf_to_doc(uploaded_file)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                splits = text_splitter.split_documents(pages)
                vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                st.session_state['retriever'] = retriever
                st.session_state['uploaded_file_hash'] = uploaded_file_hash
                st.success("준비가 완료되었습니다.")

print_messages()

init_prompt = '''
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know.
    The final answer will be in Korean.
    Question: {question} 
    Context: {context}
'''

if 'retriever' in st.session_state:
    if user_input := st.chat_input("심안봇에게 질문하세요"):

        st.chat_message("user").write(f"{user_input}")
        st.session_state["messages"].append(ChatMessage(role="user", content=user_input))

        bot_message_placeholder = st.chat_message("assistant")
        response_container = bot_message_placeholder.empty()
        stream_handler = StreamHandler(response_container)

        llm = ChatOpenAI(
            model_name="gpt-4-turbo-preview",
            temperature=0,
            streaming=True,
            callbacks=[stream_handler],
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", init_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        chain = prompt | llm

        chain_with_memory = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        context = st.session_state['retriever'].get_relevant_documents(user_input)
        response = chain_with_memory.invoke(
            {"question": user_input, "context": context},
            config={"configurable": {"session_id": session_id}},
        )

        st.session_state["messages"].append(
            ChatMessage(role="assistant", content=response.content)
        )
        response_container.markdown(response.content)
