import streamlit as st
import os
import shutil
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Universal AI Doc Assistant", layout="wide")
st.title("🤖 Universal AI Document Partner")

# Set your API Key (Best practice: use st.secrets or .env)
os.environ["GOOGLE_API_KEY"] = "API KEY"

# Define directories
DB_DIR = "chroma_db"
TEMP_DIR = "tempDir"

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# --- 2. THE CORE AI FUNCTIONS ---

def process_pdf(file_path):
    """Handles Phase 1 & 2: Loading, Splitting, and Embedding"""
    # Load
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # Embed & Store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    return vector_db

def get_rag_chain(vector_db):
    """Handles Phase 3: Setting up the Chat Brain"""
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.5)
    
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
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    return create_retrieval_chain(retriever, combine_docs_chain)

# --- 3. SIDEBAR: FILE UPLOADER ---
with st.sidebar:
    st.header("Document Settings")
    uploaded_file = st.file_uploader("Upload a PDF (Seminar, Textbook, Story)", type="pdf")
    
    if uploaded_file:
        file_path = os.path.join(TEMP_DIR, uploaded_file.name)
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("🚀 Process & Index"):
            # Clear old database to avoid mixing data
            if os.path.exists(DB_DIR):
                shutil.rmtree(DB_DIR)
            
            with st.spinner("Analyzing document..."):
                st.session_state.vector_db = process_pdf(file_path)
                st.success("Analysis Complete!")

# --- 4. MAIN CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask me anything about the document..."):
    if "vector_db" not in st.session_state:
        st.error("Please upload and process a document first!")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            chain = get_rag_chain(st.session_state.vector_db)
            response = chain.invoke({"input": prompt})
            answer = response["answer"]
            
            st.markdown(answer)
            
            # Show sources in an expandable section
            with st.expander("View Sources"):
                for doc in response["context"]:
                    st.write(f"- Page {doc.metadata.get('page')}: {doc.page_content[:200]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
