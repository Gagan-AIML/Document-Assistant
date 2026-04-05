import os
import getpass
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path ="FILE PATH"
loader=PyPDFLoader(file_path)
docs=loader.load()

text_splittre=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=text_splittre.split_documents(docs)
print(f"Total Pages: {len(docs)}")
print(f"Total Chunks: {len(chunks)}")

os.environ["GOOGLE_API_KEY"] = "API KEY"

from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector1=embeddings.embed_query(chunks[0].page_content)
vector2=embeddings.embed_query(chunks[1].page_content)
assert len(vector1) == len(vector2)
print(f"Generated embedding vector of length: {len(vector1)}")

from langchain_chroma import Chroma

Persist_directory = "chroma_db"
print(f"--- CREATING VECTOR STORE ---")
vector_db=Chroma.from_documents(chunks,embeddings,persist_directory=Persist_directory)

query="Summarize the story."
search_results=vector_db.similarity_search(query,k=1)
print(f"--- SEARCH RESULTS ---")
for i,res in enumerate(search_results):
    print(f"Result {i+1}:")
    print(res.page_content)
