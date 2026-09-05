import streamlit as st
import ollama
import base64
import io
from PIL import Image
import chromadb
from sentence_transformers import SentenceTransformer

# --- 1. ENGINE ROOM (Loading Models once) ---
st.set_page_config(page_title="Project ARIA", page_icon="🧠", layout="wide")

@st.cache_resource
def initialize_system():
    # Load the Embedding engine
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    # Connect to the permanent vector database
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="dsa_notes")
    return embedder, collection

embedder, collection = initialize_system()

# --- 2. SIDEBAR (The Controls) ---
with st.sidebar:
    st.title("⚙️ ARIA Settings")
    model_choice = st.selectbox("Reasoning Model", ["qwen2.5:3b", "qwen2.5:0.5b"])
    st.markdown("---")
    uploaded_file = st.file_uploader("Show ARIA an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Visual context active.")

# --- 3. MAIN CHAT INTERFACE ---
st.title("🧠 Project ARIA: Local AI Brain")
st.caption("Running entirely on ThinkPad T490 | No Internet Required")

# Initialize chat history for the session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. THE BRAIN LOGIC ---
if user_input := st.chat_input("What is on your mind?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Brain Processing
    with st.chat_message("assistant"):
        with st.spinner("Analyzing data and reasoning..."):
            context = ""
            vision_desc = ""

            # STAGE 1: Semantic Retrieval (Memory)
            query_vec = embedder.encode(user_input).tolist()
            results = collection.query(query_embeddings=[query_vec], n_results=1)
            if results['documents']:
                context = results['documents'][0][0]

            # STAGE 2: Visual Analysis (If image uploaded)
            if uploaded_file:
                bytes_data = uploaded_file.getvalue()
                img_str = base64.b64encode(bytes_data).decode("utf-8")
                
                vision_res = ollama.chat(
                    model='moondream',
                    messages=[{'role': 'user', 'content': 'Describe this briefly.', 'images': [img_str]}]
                )
                vision_desc = vision_res['message']['content']

            # STAGE 3: Final Multimodal Reasoning
            full_prompt = f"""
            KNOWLEDGE FROM MEMORY: {context}
            VISUAL CONTEXT: {vision_desc}
            USER QUESTION: {user_input}
            """
            
            response = ollama.chat(
                model=model_choice,
                messages=[{'role': 'user', 'content': full_prompt}],
                stream=False
            )
            
            answer = response['message']['content']
            st.markdown(answer)
            
            # Show "Metadata" for transparency
            with st.expander("🛠️ View Logic Trace"):
                st.write(f"**Retrieved Chunk:** {context[:150]}...")
                if vision_desc:
                    st.write(f"**Vision Description:** {vision_desc}")

    # Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})