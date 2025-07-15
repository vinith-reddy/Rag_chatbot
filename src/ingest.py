import os
import requests
from bs4 import BeautifulSoup
import re
import json
from urllib.parse import urlparse

# List of health document URLs and metadata
CORPUS = [
    {
        "title": "WHO Fact Sheet - Hypertension",
        "url": "https://www.who.int/news-room/fact-sheets/detail/hypertension",
        "year": 2023
    },
    {
        "title": "CDC - About Diabetes",
        "url": "https://www.cdc.gov/diabetes/about/",
        "year": None
    },
    {
        "title": "NIH MedlinePlus - Asthma",
        "url": "https://medlineplus.gov/asthma.html",
        "year": None
    },
    {
        "title": "WHO Fact Sheet - Physical Activity",
        "url": "https://www.who.int/news-room/fact-sheets/detail/physical-activity",
        "year": 2022
    },
    {
        "title": "CDC - Healthy Eating for a Healthy Weight",
        "url": "https://www.cdc.gov/healthy-weight-growth/healthy-eating/index.html",
        "year": None
    },
]

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
CLEAN_DIR = os.path.join(DATA_DIR, 'cleaned')

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

def fetch_html(url, fname):
    resp = requests.get(url)
    resp.raise_for_status()
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(resp.text)
    return resp.text

def clean_html(html, url):
    soup = BeautifulSoup(html, 'html.parser')
    # Remove nav, footer, scripts, styles, as much noise as possible
    for tag in soup(['nav', 'footer', 'script', 'style', 'header', 'form', 'aside']):
        tag.decompose()
    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, type(soup.Comment))):
        comment.extract()
    # Get main text blocks
    text = ' '.join([t.get_text(separator=' ', strip=True) for t in soup.find_all(['h1','h2','h3','h4','h5','h6','p','li'])])
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks

def ingest():
    all_chunks = []
    for doc in CORPUS:
        print(f"Processing: {doc['title']}")
        url = doc['url']
        fname = os.path.join(RAW_DIR, urlparse(url).netloc.replace('.', '_') + '_' + os.path.basename(urlparse(url).path) + '.html')
        if not os.path.exists(fname):
            html = fetch_html(url, fname)
        else:
            with open(fname, 'r', encoding='utf-8') as f:
                html = f.read()
        cleaned = clean_html(html, url)
        chunks = chunk_text(cleaned)
        for idx, chunk in enumerate(chunks):
            chunk_meta = {
                "doc_title": doc['title'],
                "doc_url": url,
                "doc_year": doc['year'],
                "chunk_id": idx,
                "text": chunk
            }
            all_chunks.append(chunk_meta)
    # Save all cleaned chunks
    with open(os.path.join(CLEAN_DIR, 'corpus_chunks.json'), 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_chunks)} chunks.")

if __name__ == "__main__":
    ingest() 