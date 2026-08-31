import os
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# 1. Setup the "Meaning Engine"
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Setup the "Permanent Storage" on your D: drive
# This creates a folder called 'chroma_db'
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="dsa_notes")

def load_and_index_files(folder_path):
    """Reads files and saves them into the Database."""
    print("📂 Reading files and creating permanent index...")
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r') as f:
                content = f.read().split('\n\n')
                for i, chunk in enumerate(content):
                    if chunk.strip():
                        # We turn the chunk into math (Vector)
                        vec = embedder.encode(chunk).tolist()
                        # We save it to ChromaDB
                        collection.add(
                            ids=[f"{filename}_{i}"],
                            embeddings=[vec],
                            documents=[chunk]
                        )
    print("✅ Indexing complete.")

def ask_aria_v2(query):
    # Step 1: Turn question into math
    query_vec = embedder.encode(query).tolist()

    # Step 2: Query the Database (The 'Search' becomes one line!)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=1
    )
    
    context = results['documents'][0][0]
    
    # Step 3: Augment and Generate (Standard RAG)
    prompt = f"Using this context: {context}\n\nQuestion: {query}"
    response = ollama.chat(model='qwen2.5:3b', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

if __name__ == "__main__":
    # Check if we need to load data or if it's already there
    if collection.count() == 0:
        load_and_index_files("knowledge")
    else:
        print(f"🧠 Brain already has {collection.count()} memories loaded.")

    # Test Question
    user_q = "Explain the Memory Search pattern in DSA."
    print(f"\n🙋 User: {user_q}")
    print(f"🧠 ARIA: {ask_aria_v2(user_q)}")