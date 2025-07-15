import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

CLEAN_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned')
INDEX_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'index')
os.makedirs(INDEX_DIR, exist_ok=True)

CHUNKS_PATH = os.path.join(CLEAN_DIR, 'corpus_chunks.json')
INDEX_PATH = os.path.join(INDEX_DIR, 'faiss.index')
META_PATH = os.path.join(INDEX_DIR, 'meta.json')

EMBED_MODEL = 'all-MiniLM-L6-v2'  # Small, fast, and effective for semantic search


def main():
    print('Loading cleaned chunks...')
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    texts = [c['text'] for c in chunks]
    print(f'Loaded {len(texts)} chunks.')

    print('Loading embedding model...')
    model = SentenceTransformer(EMBED_MODEL)
    print('Encoding chunks...')
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    print('Building FAISS index...')
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Cosine similarity (with normalized vectors)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    print(f'Saved FAISS index to {INDEX_PATH}')

    # Save metadata for mapping search results
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f'Saved metadata to {META_PATH}')

if __name__ == '__main__':
    main() 