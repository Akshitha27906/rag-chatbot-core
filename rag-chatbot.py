import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── Your API Key ─────────────────────────────────────
MY_API_KEY = "paste-api-key-here"
URL = "https://en.wikipedia.org/wiki/Artificial_intelligence"

# ── Step 1: Read Website ──────────────────────────────
def get_website_text(url):
    print(f"Reading website: {url}")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)
    print(f"Got {len(clean_text)} characters!")
    return clean_text

# ── Step 2: Split into Chunks ─────────────────────────
def split_into_chunks(text, chunk_size=500):
    print("Splitting into chunks...")
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    print(f"Created {len(chunks)} chunks!")
    return chunks

# ── Step 3: Build Retriever ───────────────────────────
def build_retriever(chunks):
    print("Building retrieval system...")
    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunks)
    print("Retrieval system ready!")
    return vectorizer, chunk_vectors

def find_relevant_chunks(question, vectorizer, chunk_vectors, chunks, top_k=3):
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, chunk_vectors)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# ── Step 4: Ask AI ────────────────────────────────────
def ask_ai(question, relevant_chunks, api_key):
    context = "\n\n".join(relevant_chunks)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    payload = {
        "model": "meta/llama-4-maverick-17b-128e-instruct",
        "messages": [{"role": "user", "content": "Use this context to answer the question.\n\nContext:\n" + context + "\n\nQuestion: " + question + "\n\nAnswer clearly and simply."}],
        "max_tokens": 512,
        "temperature": 1.00,
        "top_p": 1.00,
        "stream": False
    }
    response = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

# ── Main ──────────────────────────────────────────────
text = get_website_text(URL)
chunks = split_into_chunks(text)
vectorizer, chunk_vectors = build_retriever(chunks)

print("=" * 50)
print("  WEB RAG CHATBOT")
print("=" * 50)
print(f"I have read: {URL}")
print("Ask me anything! Type 'quit' to exit")
print("=" * 50)

while True:
    question = input("\nYou: ")
    if question.lower() == "quit":
        print("Goodbye!")
        break
    relevant = find_relevant_chunks(question, vectorizer, chunk_vectors, chunks)
    print("Thinking...")
    answer = ask_ai(question, relevant, MY_API_KEY)
    print(f"\nChatbot: {answer}")
