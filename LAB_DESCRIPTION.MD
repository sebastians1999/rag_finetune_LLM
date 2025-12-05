# Lab 2: Parameter Efficient Fine-Tuning (PEFT) of Large Language Models

**Course:** ID2223 / HT2025  
**Students:** Sebastian Schmuelling, Ramin Darudi  

---

## Overview

This project implements Parameter Efficient Fine-Tuning (PEFT) using LoRA (Low-Rank Adaptation) to fine-tune large language models on the FineTome-100k instruction dataset. The fine-tuned models are deployed in a Retrieval-Augmented Generation (RAG) chatbot interface that enables users to query documents indexed in Hopsworks Feature Store.

### Key Features

- **PEFT Fine-Tuning**: Efficient fine-tuning using LoRA with 4-bit quantization
- **Checkpoint Management**: Automatic checkpointing to HuggingFace Hub for resumable training
- **Multiple Model Support**: Fine-tuned Llama-3.2-1B and Ministral-3-3B models
- **RAG System**: Document retrieval using Hopsworks Feature Store and FAISS
- **CPU-Optimized Inference**: GGUF format models for efficient CPU deployment
- **Interactive UI**: Gradio-based chatbot with dynamic model selection

---

## Task 1: Fine-Tune a Model and Build a UI

### 1.1 Fine-Tuning Implementation

#### Models Fine-Tuned

1. **Llama-3.2-1B-Instruct**
   - Base Model: `unsloth/Llama-3.2-1B-Instruct`
   - Fine-tuned Model: `schmuelling/Llama-3.2-1B-Instruct-finetome`
   - Training Time: 30.75 minutes on A100 GPU
   - Peak Memory: 4.477 GB (11.3% of 40GB GPU)
   - Trainable Parameters: 11,272,192 (0.90% of total parameters)

2. **Ministral-3-3B-Instruct-2512**
   - Base Model: `unsloth/Ministral-3-3B-Instruct-2512`
   - Fine-tuned Model: `schmuelling/Ministral-3-3B-Instruct-2512-finetome`
   - Training Time: 429.2 minutes (~7.2 hours) on A100 GPU
   - Peak Memory: 6.896 GB (8.7% of 80GB GPU)
   - Trainable Parameters: 24,707,072 (0.64% of total parameters)

#### Fine-Tuning Process

**Dataset**: FineTome-100k (`mlabonne/FineTome-100k`)
- 100,000 instruction-following examples
- Converted from ShareGPT format to HuggingFace chat format using `standardize_sharegpt()`
- Applied model-specific chat templates (llama-3.1 for Llama, mistral for Ministral)

**Training Configuration**:
- **Framework**: Unsloth for memory-efficient fine-tuning
- **Quantization**: 4-bit (BitsAndBytesConfig with NF4 quantization and double quantization)
- **LoRA Configuration**:
  - Rank (r): 16
  - LoRA Alpha: 16
  - LoRA Dropout: 0
  - Target Modules: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- **Training Parameters**:
  - Max Sequence Length: 2048
  - Batch Size: 2 per device
  - Gradient Accumulation: 4 steps (effective batch size: 8)
  - Learning Rate: 2e-4
  - Learning Rate Scheduler: Linear decay
  - Warmup Steps: 20
  - Optimizer: AdamW 8-bit
  - Weight Decay: 0.001
  - Epochs: 1
  - Max Steps: 2000 (Llama), 12,500 (Ministral)
  - Mixed Precision: bfloat16 on Ampere+ GPUs
  - Gradient Checkpointing: Enabled ("unsloth" mode)

**Checkpointing Strategy**:
- Checkpoints saved every 100 steps (Llama) / 1000 steps (Ministral)
- Automatic push to HuggingFace Hub: `schmuelling/{model_name}-checkpoint`
- Resume training capability: Automatically detects and loads from checkpoint if repository exists
- Total checkpoint limit: 3 (oldest checkpoints automatically deleted)
- Checkpoint strategy: "checkpoint" (push on every save)

**Model Export**:
- **GGUF Format**: Multiple quantization levels exported for CPU inference
  - `q4_k_m`: 4-bit quantization (balanced quality/size)
  - `q8_0`: 8-bit quantization (higher quality)
  - `q2_k`: 2-bit quantization (smallest size)
- **Merged 4-bit**: Merged LoRA weights into 4-bit base model for HuggingFace Transformers inference
- All models pushed to HuggingFace Hub for deployment

#### Training Infrastructure

- **Platform**: Google Colab (free GPU tier)
- **GPU**: NVIDIA A100 (40GB for Llama, 80GB for Ministral)
- **Memory Efficiency**: 
  - 4-bit quantization enabled for memory reduction
  - LoRA adapters: 0.90% trainable parameters for Llama, 0.64% for Ministral
  - Gradient checkpointing enabled for additional memory savings
  - Peak memory usage: 4.477 GB (Llama), 6.896 GB (Ministral)

### 1.2 RAG System Implementation

The RAG (Retrieval-Augmented Generation) system enables the chatbot to answer questions based on indexed documents.

**Document Indexing** (`index_content.ipynb`):
- **Document Loader**: LangChain DoclingLoader for PDF processing
- **Document**: "Building Machine Learning Systems with a Feature Store.pdf"
- **Chunking Strategy**: HybridChunker with semantic chunking
- **Embeddings Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Storage**: Hopsworks Feature Store (`book_embeddings` feature group, version 1)
- **Index**: FAISS IndexFlatIP (Inner Product with L2 normalization) for similarity search
- **Indexed Chunks**: 1,333 document chunks

**Retrieval Process**:
1. User query is encoded into embedding vector using SentenceTransformer
2. FAISS performs cosine similarity search (L2-normalized inner product)
3. Top-k chunks retrieved (default: 10 chunks, configurable in `rag_prompt.yml`)
4. Context assembled with separator (`\n\n`) and passed to LLM

**RAG Prompt Template** (`prompts/rag_prompt.yml`):
- System prompt: Defines assistant role for Hopsworks documentation
- Context injection: Retrieved document chunks inserted into prompt
- Generation parameters:
  - max_tokens: 256
  - temperature: 0.7
  - stop_sequences: `["Question:", "\n\n"]`

### 1.3 User Interface

**Gradio Application** (`app.py`):
- **Model Selection**: Dropdown menus for repository and model selection
- **Dynamic Loading**: Models loaded on-demand from HuggingFace Hub using `llama-cpp-python`
- **Chat Interface**: Streaming responses with conversation history using `gr.ChatInterface`
- **Status Display**: Real-time feedback on model loading and operations
- **Model Information**: Displays model description, repository, and file details

**Features**:
- Multiple model support from different repositories (configured in `models_config.json`)
- CPU-optimized inference using `llama-cpp-python` with GGUF models
- Streaming text generation for better UX
- Example prompts for quick testing
- Error handling and user-friendly messages
- Automatic installation of `llama-cpp-python` at runtime

**Deployment**:
- Deployed to HuggingFace Spaces
- Environment variables configured via Space secrets (`HOPSWORKS_API_KEY`)
- Automatic model downloading on first load
- Supports GGUF format models for CPU inference

---

## Task 2: Improve Pipeline Scalability and Model Performance

### 2.1 Model-Centric Improvements

#### Hyperparameter Configuration

**Learning Rate Scheduling**:
- Linear learning rate decay implemented
- Warmup steps: 20 for stable training start
- Learning rate: 2e-4 (standard for LoRA fine-tuning)

**LoRA Configuration**:
- **Rank (r=16)**: Selected for balance between model capacity and parameter efficiency
- **Alpha (16)**: Set equal to rank for optimal scaling
- **Target Modules**: Selected attention and MLP layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) for maximum impact
- **Dropout**: 0 (optimized for Unsloth)

**Training Efficiency Optimizations**:
- **Gradient Accumulation (4)**: Effective batch size of 8 with minimal memory overhead
- **Mixed Precision**: bfloat16 on Ampere+ GPUs (automatic detection)
- **Optimizer**: AdamW 8-bit for memory efficiency
- **Gradient Checkpointing**: "unsloth" mode for memory savings

#### Architecture Choices

**Quantization Strategy**:
- **4-bit Quantization**: NF4 quantization with double quantization
- **Benefits**: Significant memory reduction enabling training on free Colab GPUs
- **Implementation**: BitsAndBytesConfig with `load_in_4bit=True`

**Model Selection**:
- Tested two model sizes: 1B (Llama) and 3B (Ministral) parameters
- Both models fine-tuned successfully with same LoRA configuration
- Models exported in multiple GGUF quantization levels for different use cases

### 2.2 Data-Centric Improvements

#### Dataset Used

**FineTome-100k Dataset**:
- Source: `mlabonne/FineTome-100k` from HuggingFace
- Size: 100,000 instruction-following examples
- Format: ShareGPT format (converted to HuggingFace standard format)
- Quality: High-quality, diverse instruction-following examples
- Processing: Standardized using `unsloth.chat_templates.standardize_sharegpt()`

**Data Preprocessing**:
- ShareGPT format conversion to HuggingFace chat format
- Application of model-specific chat templates
- Batched processing with `dataset.map()` for efficiency

#### Evaluation Framework

**Evaluation Script** (`evaluation/evaluate_models.py`):
- **Perplexity Calculation**: Measures model's prediction confidence
  - Lower perplexity = better model
  - Calculated on held-out test set (50 examples from FineTome-100k)
  - Compares base model vs fine-tuned model
- **Memory Efficiency Tracking**: Prints model size and parameter counts
- **Implementation**: Uses 4-bit quantization for both models during evaluation

**Evaluation Setup**:
- Base Model: `unsloth/Llama-3.2-1B-Instruct`
- Fine-tuned Model: `schmuelling/Llama-3.2-1B-Instruct-finetome`
- Test Set: 50 examples from FineTome-100k dataset
- Metrics: The finetuned model displayed an average of 2.3% improvement over base instruct model over 10 evaluation runs.

- Ministral-3 Model could not be evaluated due to how new it is and not being part of the transformer library, something we had not taken into account. 

### 2.3 Pipeline Scalability Improvements

#### Training Scalability

**Checkpoint Management**:
- Automatic checkpointing to HuggingFace Hub every 100/1000 steps
- Resume from checkpoint capability (automatic detection)
- Checkpoint versioning with limit of 3 checkpoints

**Model Versioning**:
- Versioned models on HuggingFace Hub
- Multiple quantization formats for different deployment scenarios
- Separate checkpoint and final model repositories

#### Inference Scalability

**Model Optimization**:
- GGUF quantization for CPU inference (q2_k, q4_k_m, q8_0)
- Multiple quantization levels for quality/speed trade-offs
- Merged 4-bit model for HuggingFace Transformers inference
- CPU-optimized inference using `llama-cpp-python`

**Model Loading**:
- Dynamic model loading from HuggingFace Hub
- On-demand downloading (models not stored in Space)
- Support for multiple model repositories via configuration

#### RAG System Scalability

**Index Implementation**:
- FAISS IndexFlatIP for exact similarity search
- L2 normalization for cosine similarity
- Efficient retrieval with configurable top-k

**Embedding System**:
- SentenceTransformer embeddings (`all-MiniLM-L6-v2`, 384 dimensions)
- Stored in Hopsworks Feature Store for persistence
- FAISS index built in-memory for fast retrieval

**Retrieval Configuration**:
- Configurable number of retrieved chunks (default: 10)
- Configurable context separator
- Real-time retrieval and context assembly

---

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Fine-Tuning Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│ 1. Load Base Model (4-bit quantized)                        │
│ 2. Add LoRA Adapters (r=16, alpha=16)                      │
│ 3. Load FineTome-100k Dataset                               │
│ 4. Train with Checkpointing (every 100/1000 steps)          │
│ 5. Export to GGUF Format (q2_k, q4_k_m, q8_0)              │
│ 6. Push to HuggingFace Hub                                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    RAG System                                │
├─────────────────────────────────────────────────────────────┤
│ 1. Document Indexing (DoclingLoader)                        │
│ 2. Embedding Generation (SentenceTransformers)              │
│ 3. Storage (Hopsworks Feature Store)                        │
│ 4. FAISS Index for Retrieval (IndexFlatIP)                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Inference Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│ 1. User Query                                                │
│ 2. Query Embedding (SentenceTransformer)                    │
│ 3. FAISS Retrieval (Top-k chunks)                           │
│ 4. Context Assembly                                          │
│ 5. LLM Generation (GGUF model via llama-cpp-python)         │
│ 6. Stream Response to User                                  │
└─────────────────────────────────────────────────────────────┘
```

### File Structure

```
rag_finetune_LLM/
├── app.py                          # Gradio UI application
├── models_config.json              # Model configuration
├── prompts/
│   └── rag_prompt.yml             # RAG prompt template
├── finetuning/
│   ├── Finetune_notebook_Llama.ipynb      # Llama fine-tuning
│   └── Finetune_notebook_ministral.ipynb  # Ministral fine-tuning
├── evaluation/
│   └── evaluate_models.py         # Model evaluation script
├── index_content.ipynb            # Document indexing notebook
├── requirements.txt               # Python dependencies
├── README.md                      # HuggingFace Space config
├── README_SETUP.md                # Setup instructions
└── LAB_DESCRIPTION.md             # This file
```

---

## Conclusion

This project successfully demonstrates Parameter Efficient Fine-Tuning (PEFT) using LoRA on large language models, achieving memory and computational savings while maintaining model quality. The implementation includes:

- Efficient fine-tuning with checkpointing and resume capability
- Multiple model support (Llama and Ministral)
- RAG system with Hopsworks Feature Store integration
- Production-ready UI deployed on HuggingFace Spaces
- Comprehensive documentation and evaluation framework

---

**Last Updated**: December 2025  
