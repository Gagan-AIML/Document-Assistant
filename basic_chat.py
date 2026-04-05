import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

os.environ["GOOGLE_API_KEY"] = "AIzaSyCSIUDQxkesEhqQulNnTpMQoOfDXvIoGjA"
embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_db=Chroma(persist_directory="chroma_db",embedding_function=embeddings)

llm=ChatGoogleGenerativeAI(model="gemini-3-flash-preview",temperature=0.5)

# 4. Define the System Prompt (The Rules)
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Keep the answer concise and based ONLY on the context."
    "\n\n"
    "{context}"
)

prompt=ChatPromptTemplate.from_messages(
    [ ("system", system_prompt), ("human", "{input}")]
)


# 5. Create the Retrieval Chain
# 'combine_docs_chain' handles how to pass the PDF text to the LLM
combine_docs_chain=create_stuff_documents_chain(llm=llm, prompt=prompt)

# 'retrieval_chain' connects the Vector DB to the combine_docs_chain
retrieval_chain=create_retrieval_chain(retriever=vector_db.as_retriever(search_kwargs={"k": 2}), combine_docs_chain=combine_docs_chain)

# 6. Ask a Question!
query="What is the summary and theme of the story?"
response=retrieval_chain.invoke({"input": query})

print(f"--- RESPONSE ---")
print(response["answer"])