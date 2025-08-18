
from transformers import pipeline
from sentence_transformers import SentenceTransformer
model=SentenceTransformer('all-MiniLM-L6-v2')
import numpy as np
import faiss

knowledge_base = [
  "Python is a programming language.",
  "The Earth revolves around the Sun.",
  "Transformers are great for NLP.",
  "Cats are cute animals.",
  "Neural networks learn from data."
]


knowledge_embeddings = model.encode(knowledge_base)


dimension = knowledge_embeddings.shape[1]  
index = faiss.IndexFlatL2(dimension)
index.add(np.array(knowledge_embeddings))  


def retrieve(query,top_k=2):
    query_embedding=model.encode([query])
    distances,indices=index.search(np.array(query_embedding),top_k)
    return [knowledge_base[i] for i in indices[0]]

generator = pipeline("text2text-generation",model='google/flan-t5-base')

def generate_answer(query):
    context="\n".join(retrieve(query))
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    response=generator(prompt,max_length=100)[0]['generated_text']
    return response

while True:
    query = input("You: ")
    if query.lower() in ['exit', 'quit']:
        break
    answer = generate_answer(query)
    print("Bot:", answer)
