import ollama
import time
import psutil
import os

def benchmark_inference(prompt, model_name):
    """
    Measures the performance of a specific model for a given prompt.
    """
    # Track starting state
    start_time = time.time()
    process = psutil.Process(os.getpid())
    ram_before = psutil.virtual_memory().used / (1024 * 1024) # System-wide RAM

    # The AI Inference
    try:
        response = ollama.chat(model=model_name, messages=[
            {'role': 'user', 'content': prompt}
        ])
    except Exception as e:
        return {"error": str(e)}

    # Track ending state
    end_time = time.time()
    ram_after = psutil.virtual_memory().used / (1024 * 1024)
    
    duration = end_time - start_time
    content = response['message']['content']
    
    # Calculate approximate tokens (Standard research metric: 4 chars ≈ 1 token)
    token_count = len(content) / 4
    tps = token_count / duration if duration > 0 else 0
    
    return {
        "duration": duration,
        "tps": tps,
        "output_len": len(content),
        "ram_delta": ram_after - ram_before,
        "content": content
    }

if __name__ == "__main__":
    # Models to compare for our MULTINOVA 2.0 Research Paper
    models_to_test = ['qwen2.5:3b', 'qwen2.5:0.5b']
    
    test_prompts = [
        "Explain the concept of quantum entanglement in one paragraph.",
        "Write a Python script to sort a list using bubble sort.",
        "What are the ethical implications of AI surveillance in smart cities?"
    ]

    print("\n" + "="*70)
    print("ARIA RESEARCH LAB: COMPARATIVE PERFORMANCE ANALYSIS")
    print("="*70)

    for model in models_to_test:
        print(f"\n🚀 TESTING MODEL: {model}")
        print(f"{'Input Chars':<12} | {'Time (s)':<10} | {'Tokens/sec':<10} | {'Output Chars':<15}")
        print("-" * 65)

        for p in test_prompts:
            result = benchmark_inference(p, model)
            
            if "error" in result:
                print(f"Error testing {model}: {result['error']}")
                continue

            print(f"{len(p):<12} | {result['duration']:<10.2f} | {result['tps']:<10.2f} | {result['output_len']:<15}")
            
            # Optional: Print a snippet of the code/answer to check quality
            # print(f"Snippet: {result['content'][:50]}...")

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)

    '''# Change the results line
    print(f"{len(p):<10} | {res['duration']:<10.2f} | {res['tps']:<10.2f} | {len(content):<15}")'''