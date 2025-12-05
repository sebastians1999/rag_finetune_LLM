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
import yaml
from dotenv import load_dotenv

# 1. Load Environment Variables & Validation
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not HOPSWORKS_API_KEY:
    raise ValueError("HOPSWORKS_API_KEY not found in environment variables.")

# Load models configuration
with open("models_config.json", "r") as f:
    models_config = json.load(f)

# Load RAG prompt configuration
with open("prompts/rag_prompt.yml", "r") as f:
    prompt_config = yaml.safe_load(f)

# Global variable to store the current LLM
llm = None

print("Initializing embeddings and connecting to Hopsworks...")
# 
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
def load_model(repo_name, model_name, progress=gr.Progress()):
    global llm
    try:
        progress(0, desc="Initializing...")

        # Find the repository
        repo = next((r for r in models_config["repositories"] if r["name"] == repo_name), None)
        if not repo:
            return f"Error: Repository '{repo_name}' not found in config."

        # Find the model within the repository
        model = next((m for m in repo["models"] if m["name"] == model_name), None)
        if not model:
            return f"Error: Model '{model_name}' not found in repository."

        print(f"Loading model: {model['name']}...")
        print(f"Repo: {repo['repo_id']}, File: {model['filename']}")

        progress(0.3, desc=f"Downloading/Loading {model['name']}...")

        llm = Llama.from_pretrained(
            repo_id=repo["repo_id"],
            filename=model["filename"],
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=-1,
            verbose=False
        )

        progress(1.0, desc="Complete!")
        return f"✅ Model '{model_name}' loaded successfully!"

    except Exception as e:
        llm = None
        return f"❌ Error loading model: {str(e)}"

def retrieve_context(query, k=None):
    if index is None:
        return "Error: Search index not initialized."

    # Use k from prompt config if not specified
    if k is None:
        k = prompt_config["rag"]["num_retrieved_chunks"]

    query_embedding = embeddings.encode(query).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k)

    retrieved_texts = []
    for i in indices[0]:
        if 0 <= i < len(texts):
            retrieved_texts.append(texts[i])

    # Use separator from prompt config
    separator = prompt_config["rag"]["context_separator"]

    print(f"Retrieved {len(retrieved_texts)} context chunks for the query.")
    print("Similarities:", distances)
    return separator.join(retrieved_texts)

def respond(message, history):
    """
    Generator function for streaming response.
    gr.ChatInterface passes 'message' and 'history' automatically.
    """
    if llm is None:
        yield "System Error: Models failed to load. Check console logs."
        return

    # Retrieve context using config settings
    context = retrieve_context(message)

    # Build prompt from template
    prompt = prompt_config["template"].format(
        context=context,
        question=message
    )

    # Get generation parameters from config
    gen_params = prompt_config["generation"]

    output = llm(
        prompt,
        max_tokens=gen_params["max_tokens"],
        temperature=gen_params["temperature"],
        stop=gen_params["stop_sequences"],
        stream=True
    )

    partial_message = ""
    for chunk in output:
        text_chunk = chunk["choices"][0]["text"]
        partial_message += text_chunk
        yield partial_message

with gr.Blocks(title="Hopsworks RAG ChatBot") as demo:
    gr.Markdown("<h1 style='text-align: center; color: #1EB382'>Hopsworks ChatBot</h1>")

    # Model Selection Section
    with gr.Row():
        repo_dropdown = gr.Dropdown(
            choices=[r["name"] for r in models_config["repositories"]],
            label="Select Repository",
            value=models_config["repositories"][0]["name"],
            scale=2
        )
        model_dropdown = gr.Dropdown(
            choices=[m["name"] for m in models_config["repositories"][0]["models"]],
            label="Select Model",
            value=models_config["repositories"][0]["models"][0]["name"],
            scale=2
        )
        load_button = gr.Button("Load Model", variant="primary", scale=1)

    status_box = gr.Textbox(
        label="Status",
        value="⚠️ Please select a repository and model, then click 'Load Model'",
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

    # Function to update model dropdown when repository changes
    def update_model_choices(repo_name):
        repo = next((r for r in models_config["repositories"] if r["name"] == repo_name), None)
        if repo and repo["models"]:
            model_choices = [m["name"] for m in repo["models"]]
            return gr.Dropdown(choices=model_choices, value=model_choices[0])
        return gr.Dropdown(choices=[], value=None)

    # Function to update model info display
    def update_model_info(repo_name, model_name):
        repo = next((r for r in models_config["repositories"] if r["name"] == repo_name), None)
        if not repo:
            return ""

        model = next((m for m in repo["models"] if m["name"] == model_name), None)
        if model:
            return f"**{model['name']}**\n\n{model['description']}\n\n Repository: `{repo['repo_id']}`\n\n File: `{model['filename']}`"
        return ""

    # Event handlers
    repo_dropdown.change(update_model_choices, inputs=[repo_dropdown], outputs=[model_dropdown])
    repo_dropdown.change(update_model_info, inputs=[repo_dropdown, model_dropdown], outputs=[model_info])
    model_dropdown.change(update_model_info, inputs=[repo_dropdown, model_dropdown], outputs=[model_info])
    load_button.click(load_model, inputs=[repo_dropdown, model_dropdown], outputs=[status_box])

    # Load default model info on startup
    demo.load(
        lambda: update_model_info(
            models_config["repositories"][0]["name"],
            models_config["repositories"][0]["models"][0]["name"]
        ),
        outputs=[model_info]
    )

if __name__ == "__main__":
    demo.launch(share=True)
