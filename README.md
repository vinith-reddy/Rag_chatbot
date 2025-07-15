# Health RAG Chatbot

A professional, open-source Retrieval-Augmented Generation (RAG) chatbot for health information, grounded in a curated set of trusted health documents. Built with Python, FAISS, Sentence-Transformers, and a local LLM (Llama 3-Instruct 8B via Ollama).

---

## Features
- **Healthcare-focused RAG pipeline**: Answers strictly grounded in a curated health corpus (WHO, CDC, NIH, etc.)
- **Citations**: Every answer includes inline citations to the source document and section
- **No Hallucination**: If the answer is not in the corpus, the bot says "I don’t know based on the provided corpus"
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
- If the answer is not in the corpus, the bot responds: "I don’t know based on the provided corpus"

---

## Extending
- Add more documents to the `CORPUS` list in `src/ingest.py` and re-run ingestion/vectorization
- Swap LLMs by changing the API endpoint and model in `.env`

---

## License
MIT 

---

## 1. **Sample Q&A List (for README/demo)**

Here’s a markdown-formatted sample Q&A list based on your image, ready for your documentation or demo:

```markdown
<code_block_to_apply_changes_from>
```

---

## 2. **Fixing the LLM API 404 Error**

You are still getting:
```
Error contacting LLM API: 404 Client Error: Not Found for url: http://localhost:11434/api/generate
```
This means the `/api/generate` endpoint is not available for your current Ollama model or server.

### **Checklist to Fix:**

1. **Check Ollama is Running:**
   ```bash
   ollama list
   ```
   (You already did this, so Ollama is running.)

2. **Test the Endpoint Directly:**
   Try this in your terminal:
   ```bash
   curl http://localhost:11434/api/generate -d '{
     "model": "mistral:instruct",
     "prompt": "What is asthma?",
     "stream": false
   }'
   ```
   - If you get a 404, your Ollama version or model may not support `/api/generate`.
   - If you get a valid answer, the endpoint is correct.

3. **Check Ollama Version:**
   - You are on `0.5.12`, which should support `/api/generate`.

4. **Try `/api/chat` Again:**
   Some models in newer Ollama versions only support `/api/chat`. Try:
   ```bash
   curl http://localhost:11434/api/chat -d '{
     "model": "mistral:instruct",
     "messages": [
       {"role": "user", "content": "What is asthma?"}
     ]
   }'
   ```
   - If this works, update your `.env` and code to use `/api/chat`.

5. **Check Model Name:**
   - Run `ollama list` and make sure the model name is exactly `mistral:instruct` (or whatever is listed).

---

## **What to Do Next**

- **Paste the output of the curl commands above** so I can see which endpoint works.
- **Once we confirm the working endpoint, I’ll update your code and `.env` accordingly.**

---

**Let’s get the API working, then you’ll be able to demo the Q&A with real answers and citations!**  
Let me know the curl results, or if you want me to generate a fallback (mock) Q&A output for your README/demo in the meantime.

### Sample Q&A Demo

| ID  | Sample Question                                                                                                   | Expected Answer (summary)                                                                                  |
|-----|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| Q1  | Approximately how many adults worldwide (aged 30-79) are living with hypertension according to the WHO Hypertension fact sheet? | About 1.28 billion adults (aged 30-79) have hypertension worldwide.                                       |
| Q2  | List two common symptoms of diabetes mentioned on the CDC 'About Diabetes' page.                                 | Increased thirst and frequent urination (other acceptable symptoms include blurry vision or unexplained weight loss). |
| Q3  | According to NIH MedlinePlus, what is asthma?                                                                    | A chronic disease that inflames and narrows the airways, causing wheezing, chest tightness, and shortness of breath. |
| Q4  | What is the WHO recommendation for the minimum amount of moderate-intensity physical activity per week for adults aged 18-64? | At least 150–300 minutes of moderate-intensity aerobic physical activity per week.                          |
| Q5  | Name one tip for building a healthy eating pattern from the CDC 'Healthy Eating for a Healthy Weight' page.      | Make half of your plate fruits and vegetables (other acceptable tips include choosing whole grains, drinking water instead of sugary drinks, etc.). | 