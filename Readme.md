# RAG (Retrieval-Augmented Generation) Chatbot

This repository implements a **Retrieval-Augmented Generation (RAG)** chatbot using PyTorch and other state-of-the-art libraries. The chatbot combines retrieval and generation mechanisms to answer user queries with high accuracy by leveraging pre-encoded datasets and OpenAI's inference capabilities.

---

## Features

- **Efficient Encoding**: Encodes text using `ModernBERT-base`.
- **Vector Search**: Retrieves the most relevant context using FAISS-based vector database.
- **Flexible Prompting**: Dynamically builds prompts using templates.
- **Generative Response**: Generates responses using OpenAI's API.
- **Dataset Management**: Handles dataset loading and encoding with ease.
- **Pre-computed Encoding**: Supports loading pre-computed encodings for faster startup.

---

## Requirements

- Python 3.8+
- PyTorch
- Transformers (`transformers`)
- FAISS (`faiss`)
- Datasets (`datasets`)
- tqdm
- OpenAI API

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/rag-chatbot.git
   cd rag-chatbot
   export PYTHONPATH=$(pwd)
