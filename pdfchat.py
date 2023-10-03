import os
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch


st.set_page_config(page_title="PDF Chat", page_icon="💬")
st.title("💬 PDF Chat")

uploaded_file = st.file_uploader(label="Upload PDF file", type="pdf", accept_multiple_files=True)
if not uploaded_file:
    st.info("Upload a PDF file to chat.")    

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Add your OpenAI API key to chat.")       

if uploaded_file and openai_api_key:
    def create_retriever(uploaded_file):
        # Read documents
        docs = []
        temp_dir = tempfile.TemporaryDirectory()
        for file in uploaded_file:
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            loader = PyPDFLoader(temp_filepath)
            docs.extend(loader.load())

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        # Create embeddings and store in vectordb
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

        # Define retriever
        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5})

        return retriever       

    retriever = create_retriever(uploaded_file)   

    # Setup memory for contextual conversation
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

    # Setup LLM and QA chain
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", openai_api_key=openai_api_key, temperature=0, streaming=True)
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    if len(msgs.messages) == 0:        
        msgs.add_ai_message("Do you have any question related to the PDF?")

    avatars = {"human": "user", "ai": "assistant"}
    for msg in msgs.messages:
        st.chat_message(avatars[msg.type]).write(msg.content)

    if user_query := st.chat_input(placeholder="Ask a question related to the PDF"):
        st.chat_message("user").write(user_query)        

        with st.chat_message("assistant"):                                              
            response = qa_chain.run(user_query)
            st.write(response)