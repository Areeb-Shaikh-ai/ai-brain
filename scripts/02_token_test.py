from transformers import AutoTokenizer

# The ID of the model we are building around
model_name = "Qwen/Qwen2.5-3B-Instruct"

print(f"📥 Loading the Dictionary (Tokenizer) for {model_name}...")
# This might download about 1MB of data on the first run
tokenizer = AutoTokenizer.from_pretrained(model_name)

def inspect_text(text):
    print(f"\n--- STRING: '{text}' ---")
    
    # 1. Encoding: Turning String -> List of Numbers (Token IDs)
    token_ids = tokenizer.encode(text)
    print(f"1. TOKEN IDs: {token_ids}")
    
    # 2. Decoding: Seeing the "Pieces" individually
    # We decode each ID one by one to see the 'sub-words'
    pieces = [tokenizer.decode([tid]) for tid in token_ids]
    print(f"2. THE PIECES: {pieces}")
    
    # 3. Calculation: The "Tax" on your context window
    print(f"3. TOTAL COUNT: {len(token_ids)} tokens")

if __name__ == "__main__":
    # Test 1: Your hardware
    inspect_text("ThinkPad T490")
    
    # Test 2: Complex linguistics
    inspect_text("Tokenization is undeniably fascinating.")
    
    # Test 3: Your identity
    inspect_text("Areeb")