import chainlit as cl
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
# from ctransformers import AutoModelForCausalLM
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

index_path = "book_vector_index"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  # defining embedding model

faiss_index = FAISS.load_local(
        folder_path=index_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
# allow_dangerous_deserialization=True only allow when u trust the source of the index file

# define retriever from the faiss index
retriever = faiss_index.as_retriever(search_kwargs={'k':2})

# define llm from huggingface hub

llm=CTransformers(
    model="TheBloke/Llama-2-7B-Chat-GGML",
    model_type="llama",
    max_new_tokens=600,
    temperature=0.3
)

prompt=PromptTemplate(
    input_variables=["context","question"],
    template='''
    Use ONLY the following content to answer the question:
    if relevant contet not found: say i dont know the answer 
    You must reason step by step and think carefully. Give long answers in paragraph format
    
    Context:
    {context}
    
    Question:{question}
    Answer (think step by step): '''
)

combine_documents_chain= prompt|llm

'''llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGML")
response=llm("what is a bear?")
print(response)'''
#RetrievalQA chain
from langchain.chains import RetrievalQA
# from langchain.chains.retrieval_qa import create_retrieval_chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

@cl.on_message
async def main(message: cl.Message):
    user_query = message.content

    response = qa_chain.invoke({"query": user_query})
    answer = response["result"]

    sources = ""
    for i, doc in enumerate(response["source_documents"]):
        sources += f"\n\nSource {i+1}:\n{doc.page_content[:300]}..."

    await cl.Message(content=f"**Answer:**\n{answer}\n\n---\n**Sources:**{sources}").send()

# while True:
#     query=input("Ask a question(or type exit): ")
#     if query.lower()=='exit':
#         break

#     response=qa_chain.invoke({"query":query})
#     print(f"\nAnswer : {response['result']}\n")
#     for i, doc in enumerate(response['source_documents']):
#         print(f"\nSource {i+1}:\n{doc.page_content[:300]}...")

# #return_source_documents=True means the langchain will return document chuncks that
# #  the retriever found and passed to the llm

# # let user input query
# query=input("enter your query: ")
# response=qa_chain.invoke({'query':query})

# docs = retriever.get_relevant_documents(query)
# if not docs:
#     print(" No relevant documents found in the FAISS index for your query.")
# else:
#     response = qa_chain.invoke({'query': query})
#     print('Answer:', response['result'])

#     for i, doc in enumerate(response['source_documents']):
#         print(f"\nSource {i+1}:\n{doc.page_content[:300]}...")


# # # display the answer
# # print('Answer: ',response['result']) 

# # # display the source from documents
# # for i,doc in enumerate(response['source_documents']):
# #     print(f"\nSouce {i+1} : \n{doc.page_content[:300]}...")  # print first 300 chars of reference from source


# """bcus the retrieval qa chain has :
# {
#   "result": "Backpropagation is the algorithm used to train neural networks...",  # llm generated answer
#   "source_documents": [Document(page_content="...", metadata={...}), ...]         #source_document=true
# }"""

