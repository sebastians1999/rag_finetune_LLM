import gradio as gr
import hopsworks
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from threading import Thread

# 1. Load Environment Variables & Validation
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
MODEL_ID = os.getenv("MODEL_ID", "HuggingFaceTB/SmolLM2-1.7B-Instruct")

print("Using MODEL_ID:", MODEL_ID)

if not HOPSWORKS_API_KEY:
    raise ValueError("HOPSWORKS_API_KEY not found in environment variables.")


print("Initializing models and connecting to Hopsworks...")

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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        low_cpu_mem_usage=True
    )
    
    print("Initialization complete.")

except Exception as e:
    print(f"Critical Error during initialization: {e}")
    llm = None
    tokenizer = None
    index = None

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
    if llm is None or tokenizer is None:
        yield "System Error: Models failed to load. Check console logs."
        return

    context = retrieve_context(message, k=3)

    prompt = f"""Use the following context to answer the question. If you don't know the answer, say you don't know.

Context:
{context}

Question: {message}

Answer:"""

  
    model_inputs = tokenizer([prompt], return_tensors="pt").to(llm.device)

    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    
    generation_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.1
    )

    
    thread = Thread(target=llm.generate, kwargs=generation_kwargs)
    thread.start()

   
    partial_message = ""
    
    for new_text in streamer:
        partial_message += new_text
        print("Generated so far:", partial_message)
        yield partial_message

with gr.Blocks(title="Hopsworks RAG ChatBot") as demo:
    with gr.Row():
        #gr.Image("images/hopsworks_image.jpeg", height=80, width=80, show_label=False, container=False)
        gr.Markdown("<h1 style='text-align: center'>Hopsworks RAG ChatBot</h1>")
    
    chat_interface = gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="Ask a question about your Hopsworks...", container=False, scale=7),
        examples=["What is the main topic of the documents?", "Summarize the key points."],
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch(share=True)
