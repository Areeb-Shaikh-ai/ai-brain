import ollama
import time
import psutil
import os

def benchmark_inference(prompt):
    start_time = time.time()
    
    # Track RAM usage before
    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / (1024 * 1024)
    
    response = ollama.chat(model='qwen2.5:3b', messages=[
        {'role': 'user', 'content': prompt}
    ])
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Track RAM usage after
    ram_after = process.memory_info().rss / (1024 * 1024)
    
    # Calculate approximate tokens (4 chars per token)
    content = response['message']['content']
    tokens = len(content) / 4
    tps = tokens / duration
    
    return {
        "duration": duration,
        "tokens": tokens,
        "tps": tps,
        "ram_used_mb": ram_after - ram_before,
        "response": content[:50] + "..."
    }

if __name__ == "__main__":
    prompts = [
        "Explain the concept of quantum entanglement in one paragraph.",
        "Write a Python script to sort a list using bubble sort.",
        "What are the ethical implications of AI surveillance in smart cities?"
    ]
    
    # Change the print header
    print(f"{'Prompt':<10} | {'Time (s)':<10} | {'Tokens/sec':<10} | {'Response Chars':<15}")
    print("-" * 60)

    # Change the results line
    print(f"{len(p):<10} | {res['duration']:<10.2f} | {res['tps']:<10.2f} | {len(content):<15}")