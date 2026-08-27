import os
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# 1. Initialize Tools
print("📥 Loading Meaning Engine...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 2. Block A: Load Knowledge from your D: drive folder
def load_knowledge_base(folder_path):
    all_lines = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r') as f:
                # Split by double newlines to get paragraphs/chunks
                content = f.read().split('\n\n')
                all_lines.extend([line.strip() for line in content if line.strip()])
    return all_lines

# 3. Block B: The Search Logic (Mastering the 'Find Max' Pattern)
def find_best_context(query, knowledge_base):
    query_vec = embedder.encode(query)
    best_score = -1
    best_match = ""

    print(f"🔍 Searching {len(knowledge_base)} knowledge chunks...")
    
    for text in knowledge_base:
        text_vec = embedder.encode(text)
        score = calculate_similarity(query_vec, text_vec)
        
        if score > best_score:
            best_score = score
            best_match = text
            
    return best_match, best_score

# 4. Block C: Augment & Generate (The Brain speaks with memory)
def ask_aria_with_memory(query, knowledge_base):
    # Step 1: Find the relevant info
    context, score = find_best_context(query, knowledge_base)
    
    print(f"✅ Found match (Score: {score:.4f}): '{context[:50]}...'")

    # Step 2: "Stuff" the info into the prompt
    prompt = f"""
    You are ARIA, a personal AI brain. 
    Use the provided CONTEXT to answer the user's QUESTION.
    If the context doesn't have the answer, use your own knowledge but mention that.

    CONTEXT: {context}
    QUESTION: {query}
    """
    
    # Step 3: Generate the response
    response = ollama.chat(model='qwen2.5:3b', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

if __name__ == "__main__":
    # Check if folder exists
    kb_path = "knowledge"
    if not os.path.exists(kb_path):
        print(f"❌ Error: '{kb_path}' folder not found.")
    else:
        kb = load_knowledge_base(kb_path)
        
        # Test Question
        user_q = "What are the rules for the Ends-to-Middle pointer pattern?"
        
        print(f"\n🙋 User Question: {user_q}")
        answer = ask_aria_with_memory(user_q, kb)
        
        print("\n🧠 ARIA's RESPONSE:")
        print("-" * 50)
        print(answer)
        print("-" * 50)