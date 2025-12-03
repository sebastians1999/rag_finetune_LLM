import subprocess
subprocess.run("pip install llama-cpp-python==0.3.15", shell=True, check=True)

import gradio as gr
import hopsworks
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import faiss
import numpy as np
import os
import json
from dotenv import load_dotenv

# 1. Load Environment Variables & Validation
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not HOPSWORKS_API_KEY:
    raise ValueError("HOPSWORKS_API_KEY not found in environment variables.")

# Load models configuration
with open("models_config.json", "r") as f:
    models_config = json.load(f)

# Global variable to store the current LLM
llm = None

print("Initializing embeddings and connecting to Hopsworks...")

try:
    embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    book_fg = fs.get_feature_group("book_embeddings", version=1)

    df = book_fg.read()
    
    if df.empty:
        raise ValueError("Feature group 'book_embeddings' is empty.")

    texts = df['text'].tolist()
    raw_embeddings = [emb if isinstance(emb, list) else emb.tolist() for emb in df['embedding']]
    embedding_vectors = np.array(raw_embeddings, dtype='float32')

    dimension = embedding_vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    faiss.normalize_L2(embedding_vectors)
    index.add(embedding_vectors)

    print("Embeddings and FAISS index initialized.")

except Exception as e:
    print(f"Critical Error during initialization: {e}")
    index = None

# Function to load a model dynamically
def load_model(model_name):
    global llm
    try:
        # Find the model config
        model_config = next((m for m in models_config["models"] if m["name"] == model_name), None)
        if not model_config:
            return f"Error: Model '{model_name}' not found in config."

        print(f"Loading model: {model_config['name']}...")
        print(f"Repo: {model_config['repo_id']}, File: {model_config['filename']}")

        llm = Llama.from_pretrained(
            repo_id=model_config["repo_id"],
            filename=model_config["filename"],
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=-1,
            verbose=False
        )

        return f"Model '{model_name}' loaded successfully."

    except Exception as e:
        llm = None
        return f"Error loading model: {str(e)}"

def retrieve_context(query, k=3):
    if index is None:
        return "Error: Search index not initialized."
        
    query_embedding = embeddings.encode(query).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    
    distances, indices = index.search(query_embedding, k)
    
    retrieved_texts = []
    for i in indices[0]:
        if 0 <= i < len(texts):
            retrieved_texts.append(texts[i])
            
    return "\n\n".join(retrieved_texts)

def respond(message, history):
    """
    Generator function for streaming response.
    gr.ChatInterface passes 'message' and 'history' automatically.
    """
    if llm is None:
        yield "System Error: Models failed to load. Check console logs."
        return

    context = retrieve_context(message, k=3)

    prompt = f"""Use the following context to answer the question. If you don't know the answer, say you don't know.

Context:
{context}

Question: {message}

Answer:"""


    output = llm(
        prompt,
        max_tokens=256,
        temperature=0.7,
        stop=["Question:", "\n\n"],
        stream=True  
    )
    
    partial_message = ""
    for chunk in output:
        text_chunk = chunk["choices"][0]["text"]
        partial_message += text_chunk
        yield partial_message

with gr.Blocks(title="Hopsworks RAG ChatBot") as demo:
    gr.Markdown("<h1 style='text-align: center; color: #1EB382'>Hopsworks RAG ChatBot</h1>")

    # Model Selection Section
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=[m["name"] for m in models_config["models"]],
            label="Select Model",
            value=models_config["models"][0]["name"],
            scale=3
        )
        load_button = gr.Button("Load Model", variant="primary", scale=1)

    status_box = gr.Textbox(
        label="Status",
        value="⚠️ Please load a model to start chatting",
        interactive=False
    )

    # Model info display
    model_info = gr.Markdown("")

    # Chat Interface
    chat_interface = gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(height=400),
        textbox=gr.Textbox(placeholder="Ask a question about your documents...", container=False, scale=7),
        examples=["What is the main topic of the documents?", "Summarize the key points."],
        cache_examples=False,
    )

    # Update model info when dropdown changes
    def update_model_info(model_name):
        model = next((m for m in models_config["models"] if m["name"] == model_name), None)
        if model:
            return f"**{model['name']}**\n\n{model['description']}\n\n📦 Repo: `{model['repo_id']}`\n\n📄 File: `{model['filename']}`"
        return ""

    model_dropdown.change(update_model_info, inputs=[model_dropdown], outputs=[model_info])
    load_button.click(load_model, inputs=[model_dropdown], outputs=[status_box])

    # Load default model info on startup
    demo.load(lambda: update_model_info(models_config["models"][0]["name"]), outputs=[model_info])

if __name__ == "__main__":
    demo.launch(share=True)
