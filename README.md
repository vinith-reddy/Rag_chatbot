# Health RAG Chatbot

A simple, open-source Retrieval-Augmented Generation (RAG) chatbot for health information, grounded in a curated set of trusted health documents. Built with Python, FAISS, Sentence-Transformers, and a local LLM (Llama 3-Instruct 8B via Ollama).

---

## Features
- **Healthcare-focused RAG pipeline**: Answers strictly grounded in a curated health corpus (WHO, CDC, NIH, etc.)
- **Citations**: Every answer includes inline citations to the source document and section
- **No Hallucination**: If the answer is not in the corpus, the bot says "I'm sorry, but I can only provide information based on the health-related documents in my knowledge base. Please ask a question related to healthcare, and I’d be happy to assist!"
- **Minimal, modern UI**: Streamlit app with a dark chat-style interface
- **Open-source LLM**: Uses Mistral-Instruct (via Ollama)
- **Easy extensibility**: Add more documents or swap LLMs with minimal changes

---

## Setup

### 1. Clone the repo and create a conda environment
```bash
conda create -n RCB python=3.10 -y
conda activate RCB
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download and run Mistral-Instruct with Ollama
- [Install Ollama](https://ollama.com/download)
- Pull the model:
  ```bash
  ollama pull mistral:instruct
  ollama run mistral:instruct
  ```
- The default API endpoint is `http://localhost:11434/api/generate`

### 4. Configure environment variables
Create a `.env` file in the project root:
```
LLAMA3_API_URL=http://localhost:11434/api/generate
LLAMA3_MODEL=mistral:instruct
```

### 5. Ingest and index the health corpus
```bash
python src/ingest.py
python src/vectorize.py
```

### 6. Run the Streamlit app
```bash
streamlit run src/app.py
```

---

## Usage
- Enter a health-related question in the chat box.
- The bot retrieves the most relevant passages from the corpus and synthesizes an answer, always with citations.
- If the answer is not found in the corpus, a disclaimer is shown.

---

## Architecture
- **Ingestion**: Downloads, cleans, and chunks health documents (HTML)
- **Vectorization**: Embeds chunks with Sentence-Transformers and indexes with FAISS
- **Retrieval**: Finds top-k relevant chunks for a query
- **LLM Synthesis**: Local Mistral-Instruct (via Ollama) generates answers, strictly grounded in retrieved context
- **UI**: Streamlit app for user interaction

---

## Citation & Grounding Policy
- All answers are generated strictly from retrieved passages
- Inline citations reference the document and section (e.g., [WHO Hypertension, 2023])
- If the answer is not in the corpus, the bot responds: "I'm sorry, but I can only provide information based on the health-related documents in my knowledge base. Please ask a question related to healthcare, and I’d be happy to assist!"

---

## Extending
- Add more documents to the `CORPUS` list in `src/ingest.py` and re-run ingestion/vectorization
- Swap LLMs by changing the API endpoint and model in `.env`

---

## License
MIT 


### Sample Q&A Demo

| ID  | Sample Question                                                                                                   | Expected Answer (summary)                                                                                  |
|-----|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| Q1  | Approximately how many adults worldwide (aged 30-79) are living with hypertension according to the WHO Hypertension fact sheet? | About 1.28 billion adults (aged 30-79) have hypertension worldwide.                                       |
| Q2  | List two common symptoms of diabetes mentioned on the CDC 'About Diabetes' page.                                 | Increased thirst and frequent urination (other acceptable symptoms include blurry vision or unexplained weight loss). |
| Q3  | According to NIH MedlinePlus, what is asthma?                                                                    | A chronic disease that inflames and narrows the airways, causing wheezing, chest tightness, and shortness of breath. |
| Q4  | What is the WHO recommendation for the minimum amount of moderate-intensity physical activity per week for adults aged 18-64? | At least 150–300 minutes of moderate-intensity aerobic physical activity per week.                          |
| Q5  | Name one tip for building a healthy eating pattern from the CDC 'Healthy Eating for a Healthy Weight' page.      | Make half of your plate fruits and vegetables (other acceptable tips include choosing whole grains, drinking water instead of sugary drinks, etc.). | 