import os
import time
import base64
import io
from PIL import Image
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# 1. Initialize Senses and Memory
print("🧠 Initializing Unified Brain...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="dsa_notes")

def encode_image(image_path):
    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def unified_inference(image_path, user_query):
    start_total = time.time()

    # --- STAGE 1: VISION ---
    print("\n👁️ STAGE 1: Analyzing Image...")
    img_str = encode_image(image_path)
    
    start_vision = time.time()
    vision_res = ollama.chat(
        model='moondream',
        messages=[{'role': 'user', 'content': 'Describe this image briefly.', 'images': [img_str]}]
    )
    image_description = vision_res['message']['content']
    vision_time = time.time() - start_vision
    print(f"✅ Vision complete ({vision_time:.2f}s): {image_description[:50]}...")

    # --- STAGE 2: RETRIEVAL ---
    print("\n🔍 STAGE 2: Searching Long-term Memory...")
    # We search based on the IMAGE description + the USER query
    search_query = f"{image_description} {user_query}"
    query_vec = embedder.encode(search_query).tolist()
    
    results = collection.query(query_embeddings=[query_vec], n_results=1)
    context = results['documents'][0][0]
    print(f"✅ Memory retrieved: {context[:50]}...")

    # --- STAGE 3: REASONING ---
    print("\n🧠 STAGE 3: Final Reasoning...")
    prompt = f"""
    IMAGE DESCRIPTION: {image_description}
    RELEVANT NOTES: {context}
    
    USER QUESTION: {user_query}
    
    Answer the question by connecting the image and the notes.
    """
    
    start_reasoning = time.time()
    final_res = ollama.chat(model='qwen2.5:3b', messages=[{'role': 'user', 'content': prompt}])
    reasoning_time = time.time() - start_reasoning
    
    total_time = time.time() - start_total
    
    return {
        "answer": final_res['message']['content'],
        "metrics": {
            "vision_time": vision_time,
            "reasoning_time": reasoning_time,
            "total_time": total_time
        }
    }

if __name__ == "__main__":
    # Test with the train image and a question about patterns
    img = "test_image.png" 
    query = "Does the object in this image follow any DSA patterns like Parallel Pointers?"
    
    if os.path.exists(img):
        result = unified_inference(img, query)
        print("\n" + "="*50)
        print("FINAL BRAIN OUTPUT:")
        print("-" * 50)
        print(result['answer'])
        print("="*50)
        print(f"METRICS: Vision: {result['metrics']['vision_time']:.2f}s | Reasoning: {result['metrics']['reasoning_time']:.2f}s | Total: {result['metrics']['total_time']:.2f}s")
    else:
        print(f"❌ Please put '{img}' in the folder.")