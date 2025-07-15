# Health RAG Chatbot: Architecture & Design

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline for a health information assistant. The system answers user queries strictly based on a curated set of health documents, always citing sources and refusing to hallucinate.

---

## Pipeline Components

```
User Query
   |
   v
[Streamlit UI]
   |
   v
[RAG Pipeline]
   |
   |---> [Retriever: FAISS + Sentence-Transformers]
   |         |
   |         v
   |   Top-k relevant chunks
   |
   v
[LLM Synthesis: Mistral-Instruct (Ollama, /api/generate)]
   |
   v
[Grounded Answer + Citations]
   |
   v
[Streamlit UI]
```

---

## Component Details
- **Ingestion**: Downloads, cleans, and chunks HTML health documents.
- **Vectorization**: Embeds chunks using Sentence-Transformers; stores in FAISS for fast similarity search.
- **Retrieval**: For each query, retrieves top-k most relevant chunks using cosine similarity.
- **LLM Synthesis**: Mistral-Instruct (via Ollama, /api/generate) is prompted with the retrieved context and instructed to only answer from the context, always with citations. If the answer is not found, it must say "I don’t know based on the provided corpus."
- **UI**: Streamlit app with a modern, dark chat interface.

---

## Design Decisions & Trade-offs
- **Open-source stack**: Uses only free/open-source tools (FAISS, Sentence-Transformers, Ollama, Streamlit).
- **Strict grounding**: LLM is never allowed to answer outside the retrieved context, minimizing hallucination risk.
- **Minimal UI**: Focused on speed and clarity for demo purposes.
- **Extensibility**: New documents can be added with minimal reconfiguration.
- **Performance**: Fast retrieval and response for small/medium corpora; can be scaled with more powerful hardware or distributed vector DBs.

---

## Future Improvements
- Add hybrid retrieval (keyword + vector) for better accuracy
- Support for PDF and other document formats
- Real-time chat history and user feedback
- Cloud deployment for scalability
- More advanced LLM guardrails (e.g., answer validation)
- Automated document ingestion pipelines

---

## References
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence-Transformers](https://www.sbert.net/)
- [Ollama](https://ollama.com/)
- [Streamlit](https://streamlit.io/) 