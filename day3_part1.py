import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader=PyPDFLoader("deep_learning_fleuret.pdf")
documents=loader.load()   # creates one document per page

splitter=RecursiveCharacterTextSplitter(
    chunk_size=300,       #max_length of each chunk
    chunk_overlap=50,       # sliding window overlap
    separators=['\n\n','\n','.',' ','']
)

chunked_docs=splitter.split_documents(documents)

for doc in chunked_docs:
    doc.page_content='passage:'+doc.page_content

'''intfloat/e5-base-v2 was trained to distinguish between queries and passages using specific prefixes:

    "query: " for questions

    "passage: " for documents

Adding "passage: " ensures the embeddings match how the model learned to represent 
relevant text. Without it, retrieval accuracy drops because the model can’t properly 
align queries with passages.'''

# now we Convert Chunks to Vectors Using Embedding Model + Store in FAISS
# embedding model is the same as we used for sentence _transformers

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
#chroma db
"""dont use all-MiniLM-L6-v2 try any other
alternative : all-MiniLM-L12-v2
              sentence-t5-base
              intfloat/e5-base-v2"""
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  # defining embedding model
# now we create and save faiss index locally

index_path = "book_vector_index"


faiss_index = FAISS.from_documents(chunked_docs, embedding_model)
faiss_index.save_local(index_path)
    
