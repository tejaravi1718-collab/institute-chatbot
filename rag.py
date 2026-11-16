import os
import sys
from pathlib import Path

import faiss
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

DATA_DIR = Path("data")
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"
TOP_K = 3

SYSTEM_PROMPT = (
    "You are the friendly admissions assistant for a coaching institute. "
    "Answer the question using ONLY the context provided. "
    "If the answer is not in the context, say you don't have that information "
    "and suggest contacting the institute office. "
    "Never invent fees, dates, timings or phone numbers. "
    "Keep answers short and clear."
)


def load_chunks():
    """Read every .txt file in data/ and split it into sections."""
    chunks = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for block in text.split("\n\n"):
            block = block.strip()
            if len(block) > 20:
                chunks.append({"text": block, "source": path.name})
    return chunks


def build_index():
    """Turn each section into numbers and store them in a searchable index."""
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError("No .txt files found in the data folder.")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        [c["text"] for c in chunks], normalize_embeddings=True
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return model, index, chunks


def retrieve(question, model, index, chunks):
    """Find the sections of the brochure closest to the question."""
    q = model.encode([question], normalize_embeddings=True).astype("float32")
    _, ids = index.search(q, TOP_K)
    return [chunks[i] for i in ids[0] if i != -1]


def answer(question, model, index, chunks):
    """Retrieve relevant sections, then ask the AI to answer from them."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found. Check your .env file.")

    found = retrieve(question, model, index, chunks)
    context = "\n\n---\n\n".join(c["text"] for c in found)

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.2,
        max_completion_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content, found


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("Loading... (the first run downloads a small model, please wait)")
    model, index, chunks = build_index()
    print(f"Ready. Loaded {len(chunks)} sections. Type 'exit' to stop.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        reply, found = answer(question, model, index, chunks)
        print(f"\nBot: {reply}")
        print("   [sections used: " + " | ".join(
            c["text"].splitlines()[0][:40] for c in found) + "]\n")