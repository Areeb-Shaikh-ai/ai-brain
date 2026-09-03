import ollama
import base64
from PIL import Image
import io

def test_vision(image_path):
    # 1. Open and verify the image
    try:
        img = Image.open(image_path)
        # --- THE FIX ---
        # If the image has transparency (RGBA), convert it to standard RGB
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        # ----------------
        print(f"✅ Image loaded and converted: {img.size}")
    except Exception as e:
        print(f"❌ Could not open image: {e}")
        return

    # 2. Convert Image to Base64
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # 3. Send to the Brain
    print("🧠 ARIA is looking at the image... (Inference starting)")
    
    response = ollama.chat(
        model='moondream',
        messages=[{
            'role': 'user',
            'content': 'Describe this image in detail, including the colors and the background.',
            'images': [img_str]
        }]
    )

    print("\n--- BRAIN RESPONSE ---")
    print(response['message']['content'])

if __name__ == "__main__":
    # IMPORTANT: Ensure this filename matches the file in your folder!
    # If your file is a png, change it to "test_image.png"
    test_vision("test_image.png")