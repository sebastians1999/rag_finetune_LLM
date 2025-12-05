# Setup and Deployment Guide

This guide walks you through setting up and deploying the Hopsworks RAG ChatBot to HuggingFace Spaces.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Setup](#local-setup)
3. [Indexing Documents](#indexing-documents)
4. [Configuring Models](#configuring-models)
5. [Deploying to HuggingFace Spaces](#deploying-to-huggingface-spaces)
6. [Syncing with GitHub](#syncing-with-github)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10** installed locally
- **Git** installed
- **Hopsworks Account**: Sign up at [hopsworks.ai](https://www.hopsworks.ai/)
- **HuggingFace Account**: Sign up at [huggingface.co](https://huggingface.co/)
- **PDF Documents** you want to index for RAG

---

## Local Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd rag_finetune_LLM
```

### 2. Create Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
# .env
HOPSWORKS_API_KEY=your_hopsworks_api_key_here
```

**Get your Hopsworks API Key:**
1. Go to [Hopsworks](https://www.hopsworks.ai/)
2. Navigate to your project
3. Click on your profile → Settings → API Keys
4. Create a new API key and copy it

---

## Indexing Documents

### 1. Add Your PDF Document

Place your PDF file in the project directory (e.g., `content/your_content.pdf`)

### 2. Update the Indexing Notebook

Open `index_content.ipynb` and update the PDF path:

```python
PDF_PATH = "content/your_content.pdf"  # Update this
```

### 3. Run the Notebook

Execute all cells in `index_content.ipynb`:

```bash
jupyter notebook index_content.ipynb
```

This will:
- Load and chunk your PDF using Docling
- Generate embeddings with sentence-transformers
- Upload to Hopsworks Feature Store as `content` feature group

**Note:** This only needs to be done once. The embeddings will be available for all deployments.

---

## Configuring Models

### 1. Edit Model Configuration

Update `models_config.json` with your models.



### 2. Model Format Requirements

- Models should be in **GGUF format** (for CPU-optimized inference unless you have GPUs)
- Hosted on HuggingFace Hub

---

## Deploying to HuggingFace Spaces

### Method 1: Direct Git Push (Recommended)

#### 1. Create a New Space

1. Go to [HuggingFace Spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Configure:
   - **Name**: `your-rag-chatbot`
   - **SDK**: Gradio
   - **Hardware**: CPU basic (free tier works fine)
   - **Visibility**: Public or Private

#### 2. Get Your HuggingFace Token

1. Go to [HuggingFace Settings → Tokens](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Give it a name (e.g., "spaces-deploy")
4. Select **Write** permission
5. Copy the token

#### 3. Connect Your Repository

```bash
# Add HuggingFace Space as remote
```bash
git remote add space https://YOUR_USERNAME:YOUR_TOKEN@huggingface.co/spaces/your-username/your-rag-chatbot
```

#### 4. Configure Secrets

In your Space settings on HuggingFace:

1. Go to **Settings** → **Repository secrets**
2. Add the following secret:
   - **Name**: `HOPSWORKS_API_KEY`
   - **Value**: Your Hopsworks API key

#### 5. Wait for Build

The Space will automatically build and deploy. This may take a couple of minutes.

---

### Method 2: GitHub Sync (Automatic)

#### 1. Enable GitHub Actions

The repository includes `.github/workflows/sync_to_huggingface.yaml` for automatic syncing.

#### 2. Add GitHub Secrets

In your GitHub repository:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add:
   - **Name**: `HF_TOKEN`
   - **Value**: Your HuggingFace write token

**Get your HuggingFace Token:**
1. Go to [HuggingFace Settings → Tokens](https://huggingface.co/settings/tokens)
2. Create a new token with **write** permissions
3. Copy the token

#### 3. Update Workflow File (if needed)

Edit `.github/workflows/sync_to_huggingface.yaml` and update:

```yaml
env:
          HF_TOKEN: ${{ secrets.HUGGINGFACE_SYNC_TOKEN }} #leave this
          HF_SPACE_URL: https://huggingface.co/spaces/your-username/your-space-name
```

#### 4. Automatic Syncing

Now, every push to your `main` branch will automatically sync to HuggingFace Spaces!

```bash
git add .
git commit -m "Update model configuration"
git push origin main  # Automatically syncs to HF Spaces
```

---

## Testing

### Local Testing

Before deploying, test locally:

```bash
python app.py
```

This will:
1. Install llama-cpp-python at runtime
2. Connect to Hopsworks and load embeddings
3. Launch Gradio interface at a local host (exact url can be found in the command line)

### Testing on HuggingFace Spaces

1. Open your Space URL: `https://huggingface.co/spaces/your-username/your-space-name`
2. Select a model from the dropdown
3. Click **"Load Model"** (wait 1-3 minutes for first load)
4. Once loaded, ask a question related to your documents
5. Verify the response uses context from your indexed documents

---

## Configuration Reference

### README.md (Space Configuration)

### models_config.json

Defines available models in the dropdown:

```json
{
  "models": [
    {
      "name": "Display Name",           // Shown in dropdown
      "repo_id": "username/repo",       // HuggingFace model repository
      "filename": "model.gguf",         // GGUF file in the repo
      "description": "Model description" // Shown in UI
    }
  ]
}
```


**Happy Deploying! 🚀**
