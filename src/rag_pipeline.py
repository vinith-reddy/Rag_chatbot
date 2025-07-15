import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

INDEX_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'index')
INDEX_PATH = os.path.join(INDEX_DIR, 'faiss.index')
META_PATH = os.path.join(INDEX_DIR, 'meta.json')
EMBED_MODEL = 'all-MiniLM-L6-v2'

# Ollama API config
LLAMA3_API_URL = os.environ.get('LLAMA3_API_URL', 'http://localhost:11434/api/generate')
LLAMA3_MODEL = os.environ.get('LLAMA3_MODEL', 'mistral:instruct')

TOP_K = 3

def load_index_and_meta():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return index, meta

def embed_query(query, model):
    emb = model.encode([query], normalize_embeddings=True)
    return emb.astype(np.float32)

def retrieve(query, model, index, meta, top_k=TOP_K):
    q_emb = embed_query(query, model)
    D, I = index.search(q_emb, top_k)
    results = [meta[i] for i in I[0]]
    return results

def format_context(chunks):
    context = ""
    for c in chunks:
        citation = f"[{c['doc_title']}{', ' + str(c['doc_year']) if c['doc_year'] else ''}]"
        context += f"{c['text']} {citation}\n"
    return context

def call_ollama_generate_api(prompt):
    data = {
        "model": LLAMA3_MODEL,
        "prompt": prompt,
        "stream": False
    }
    resp = requests.post(LLAMA3_API_URL, json=data)
    resp.raise_for_status()
    result = resp.json()
    return result.get("response", "[Error: Unexpected LLM response format]")

def answer_query(query):
    index, meta = load_index_and_meta()
    model = SentenceTransformer(EMBED_MODEL)
    top_chunks = retrieve(query, model, index, meta)
    context = format_context(top_chunks)
    prompt = f"You are a healthcare assistant. Only answer using the provided context. If the answer is not in the context, say 'I don’t know based on the provided corpus.' Always cite sources inline.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer (with citations):"
    try:
        answer = call_ollama_generate_api(prompt)
    except Exception as e:
        answer = f"Error contacting LLM API: {e}"
    return answer, top_chunks

if __name__ == '__main__':
    # Example usage
    q = input("Enter your health question: ")
    answer, chunks = answer_query(q)
    print("\nAnswer:\n", answer)
    print("\nRetrieved Chunks:")
    for c in chunks:
        print(f"- {c['doc_title']} (chunk {c['chunk_id']})") 