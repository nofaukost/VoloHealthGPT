# VoloHealthGPT

This repository contains an AI-powered psychologist chatbot that uses advanced language models and retrieval-augmented generation (RAG) to provide empathetic and informative responses to users seeking psychological support. The chatbot utilizes the Meta's Llama 3.1 8B model for language processing and the Nomic Embed Text model for embeddings, both through Ollama.

## Project Structure

```
.
├── data/
│   └── knowledge base files (pdf, doc, txt, md, etc)
├── notebooks/
│   └── db-populate.ipynb
├── src/
│   ├── chatbot.py
│   └── rag-chatbot.py
├── requirements.txt
└── README.md
```

## Setup

This project is designed to run on an AWS g5.4xlarge instance using Miniconda with Python 3.12 and Docker. Follow these steps to set up the environment:

1. Clone this repository:
   ```
   git clone https://github.com/your-username/ai-psychologist-chatbot.git
   cd ai-psychologist-chatbot
   ```

2. Install Miniconda if you haven't already:
   ```
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

3. Install Docker if not already installed:
   ```
   sudo apt-get update
   sudo apt-get install docker.io
   ```

4. Start the Ollama Docker container:
   ```
   docker run -d --gpus=all -v /home/ollama:/root/.ollama:z -p 11434:11434 --name ollama ollama/ollama
   ```

5. Pull the required Ollama models:
   ```
   docker exec -it ollama ollama pull llama3.1:8b
   docker exec -it ollama ollama pull nomic-embed-text
   ```

6. Create and activate a new Conda environment:
   ```
   conda create -n psychochat python=3.12
   conda activate psychochat
   ```

7. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Populating the Knowledge Base

Before running the chatbot, you need to populate the knowledge base:

1. Place your knowledge base files (PDF, DOC, TXT, MD, etc.) in the `data/` directory.
2. Open and run the `notebooks/db-populate.ipynb` Jupyter notebook:
   ```
   jupyter notebook notebooks/db-populate.ipynb
   ```
   This notebook will process the files in the `data/` directory and create a vector database for use with the RAG system.

### Running the Chatbot

There are two versions of the chatbot available:

1. Basic Chatbot:
   ```
   python src/chatbot.py
   ```
   This version uses llama3.1:8b model's knowledge with prompt engineering without any RAG.

2. RAG-enhanced Chatbot:
   ```
   python src/rag-chatbot.py
   ```
   This version uses Retrieval-Augmented Generation to provide responses based on the knowledge base in addition.

Both chatbots will start a Streamlit web application. Open the provided URL in your web browser to interact with the chatbot.

## Features

- Utilizes the Llama 3.1 8B model through Ollama for natural language processing
- Uses the Nomic Embed Text model for generating embeddings
- Classification of user input into various psychological categories (Using llama3.1:8b)
- Empathetic and supportive responses tailored to the user's needs via prompt engineering
- Retrieval-Augmented Generation for more informed and context-aware responses (in rag-chatbot.py)
- Streamlit-based user interface for easy interaction
- Source citation and display for transparency (in rag-chatbot.py)

## TODO List

We have several plans to improve and expand this project:

1. Improve the knowledge base (KB) with more comprehensive and diverse psychological resources.
2. Add summarization of the knowledge base and improve metadata to enhance retrieval accuracy.
3. Apply hybrid search techniques to improve the match quality between user queries and KB content.
4. Add reranking to further refine the relevance of retrieved information.
5. Improve the prompts used for both classification and response generation to enhance the chatbot's effectiveness and empathy.

## Disclaimer

This Chatbot is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## License

MIT License


