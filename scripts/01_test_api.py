import requests
import json

# This is the "address" where Ollama is listening on your ThinkPad
OLLAMA_URL = "http://localhost:11434/api/chat"

def send_signal_to_brain(question):
    payload = {
        "model": "qwen2.5:3b",
        "messages": [
            {"role": "system", "content": "You are a helpful AI brain running on a ThinkPad T490."},
            {"role": "user", "content": question}
        ],
        "stream": False # We want the full answer at once for this test
    }
    
    try:
        print(f"📡 Sending question: {question}")
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        
        # Pull the answer out of the technical response
        answer = response.json()['message']['content']
        print("\n🧠 BRAIN RESPONSE:")
        print("-" * 30)
        print(answer)
        print("-" * 30)
        
    except Exception as e:
        print(f"❌ Connection Failed! Error: {e}")
        print("💡 Make sure Ollama is running in your taskbar!")

if __name__ == "__main__":
    # Feel free to change this question!
    send_signal_to_brain("Tell me one cool thing about Artificial Intelligence.")