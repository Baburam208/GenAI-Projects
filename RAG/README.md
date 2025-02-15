## This is a RAG project

Create `.env` file to store the api key.

Sample `.env` file contains api key as follows.

```
OPENAI_API_KEY = "Your-api-key-here"
GOOGLE_API_KEY = "Your-api-key-here"
HUGGINGFACEHUB_ACCESS_TOKEN = "Your-api-key-here"
```

### Tools used
1. We have used Gemini chat model `ChatGoogleGenerativeAI` with model `gemini-1.5-pro`.
2. For vector store we have used FAISS (Facebook AI Similarity Serach).
3. The external source of knowledge is a pdf of paper.
4. For query and external text chunk embedding we have used `HuggingFaceEmbeddings`.
5. And the framework used for building RAG system is `LangChain`.

### Steps
#### 1
```
# Use a Different Cache Directory for caching `huggingface_hub`
os.environ["HF_HOME"] = r"D:\GenAI-Project-env\RAG"
```
This is used to store the cache files, in the defined directory location.

#### 2
```
llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0)
```
This initializes the Gemini chat model, which will be used to generate the fluent response based on the query and retrieved chunk texts.

#### 3
```
# load the document
loader = UnstructuredLoader(r"./content/stgcn.pdf")  # Updated class
documents = loader.load()
```
This loads the external knowledge base.

#### 4
After loading the external knowledge base, we have to chunk the text, as follows:
```
# create text chunks
text_splitter = CharacterTextSplitter(separator='/n',
                                      chunk_size=1000,
                                      chunk_overlap=200)

text_chunks = text_splitter.split_documents(documents)
```
This code snippets splits the knowledge base based on newline and the chunk size is set to 1000, with chunk overlap of 200 to make chunks 
hold context from earlier chunks.

#### 5
After chunking the text chunks are embedded using `HuggingFaceEmbeddings`.
```
# loading the vector embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") 
# vector embedding for text chunks
knowledge_base = FAISS.from_documents(text_chunks, embeddings)
```
FAISS vector store is stored in memory (RAM) during runtime.
#### 6
Then finally, retrieving external data source and responsing to the query by llm is done as follows.
```
# chain for qa retrieval
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=knowledge_base.as_retriever())

question = "What is this document about, can you explain a little bit more?"
response = qa_chain.invoke({"query": question})
print(response["result"])
```

#### Response screen shot

![alt text](<assets/response ss.png>)

The response seems not quite good and clear. Maybe, because of the language model or other un-tuned hyperparameteres.