import os
import logging

# Suppress INFO logging messages
logging.basicConfig(level=logging.WARNING)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_unstructured import UnstructuredLoader  # Updated import
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

load_dotenv()

# Use a Different Cache Directory for caching `huggingface_hub`
os.environ["HF_HOME"] = r"D:\GenAI-Project-env\RAG"

llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0)

# load the document
loader = UnstructuredLoader(r"./content/stgcn.pdf")  # Updated class
documents = loader.load()

# create text chunks
text_splitter = CharacterTextSplitter(separator='/n',
                                      chunk_size=1000,
                                      chunk_overlap=200)

text_chunks = text_splitter.split_documents(documents)

# loading the vector embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") 
# vector embedding for text chunks
knowledge_base = FAISS.from_documents(text_chunks, embeddings)

# chain for qa retrieval
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=knowledge_base.as_retriever())

question = "What is this document about, can you explain a little bit more?"
response = qa_chain.invoke({"query": question})
print(response["result"])
