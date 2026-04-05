import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st
import shutil
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="InsightArchive", layout="wide")
st.title("🤖 InsightArchive: Intelligent Document Explorer")

# SECRETS MANAGEMENT: 
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    # For local testing, ensure it's in your environment
    os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE" 

DB_DIR = "chroma_db"
TEMP_DIR = "tempDir"

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# --- 2. THE CORE AI FUNCTIONS ---

def process_document(file_path):
    # Choose loader based on file extension
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx") or file_path.endswith(".doc"):
        loader = Docx2txtLoader(file_path)
    else:
        st.error("Unsupported file format!")
        return None

    docs = loader.load()
    
    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # Embed & Store (Using latest stable embedding model)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    return vector_db

def get_rag_chain(vector_db):
    """Handles Phase 3: Setting up the Chat Brain"""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
    
    system_prompt = (
        "You are an expert assistant. Use the provided context to answer the question. "
        "Generate a detailed and structured response based ONLY on the context. "
        "If you don't know, say you don't know."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    return create_retrieval_chain(retriever, combine_docs_chain)

# --- 3. SIDEBAR: FILE UPLOADER ---
with st.sidebar:
    st.header("Document Settings")
    # Updated to accept docx and doc
    uploaded_file = st.file_uploader("Upload a Document (PDF or Word)", type=["pdf", "docx", "doc"])
    
    if uploaded_file:
        file_path = os.path.join(TEMP_DIR, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("🚀 Process & Index"):
            if os.path.exists(DB_DIR):
                shutil.rmtree(DB_DIR)
            
            with st.spinner("Analyzing document..."):
                st.session_state.vector_db = process_document(file_path)
                st.success(f"Analysis Complete: {uploaded_file.name}")

# --- 4. MAIN CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about the document..."):
    if "vector_db" not in st.session_state:
        st.error("Please upload and process a document first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            chain = get_rag_chain(st.session_state.vector_db)
            response = chain.invoke({"input": prompt})
            answer = response["answer"]
            
            st.markdown(answer)
            
            with st.expander("View Sources"):
                for doc in response["context"]:
                    page_label = doc.metadata.get('page', 'N/A')
                    st.write(f"- Source Fragment (Page {page_label}): {doc.page_content[:200]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})