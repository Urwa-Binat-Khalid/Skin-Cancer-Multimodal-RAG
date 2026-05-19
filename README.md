# Skin Cancer Multimodal RAG

A multimodal Retrieval-Augmented Generation system for skin cancer research, medical evidence retrieval, dermoscopic image understanding, and clinical question answering.

---

# Overview

This project combines medical literature, dermoscopic image datasets, vector databases, knowledge graphs, and large language models to build an evidence-based skin cancer question-answering system.

The system retrieves relevant medical evidence from indexed sources and generates grounded responses using Retrieval-Augmented Generation instead of relying only on the language model.

---

# Key Features

- Medical literature ingestion from PubMed and PMC
- Dermoscopic image dataset indexing
- Text preprocessing and chunking
- Dense embedding generation
- Multimodal text-image retrieval
- Qdrant vector database integration
- Knowledge graph construction
- Hybrid retrieval using semantic search, BM25, and graph retrieval
- LLM-based grounded medical responses
- Modular FastAPI-ready architecture
- Evaluation pipeline for retrieval and answer quality

---

# Project Structure

```text
Skin-Cancer-Multimodal-RAG/
│
├── config.py
├── data_ingestion.py
├── document_processor.py
├── embeddings.py
├── vector_store.py
├── knowledge_graph.py
├── retrieval.py
├── llm.py
├── multimodal.py
├── evaluation.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── notebooks/
│   └── medical-rag (1).ipynb
```

---

# Modules

## config.py

Contains project configuration, API keys, model names, paths, database settings, and global constants.

## data_ingestion.py

Handles data collection and loading from medical literature sources and dermoscopic image datasets.

## document_processor.py

Processes medical documents, extracts useful text, cleans content, and creates chunks for retrieval.

## embeddings.py

Generates text and image embeddings using transformer-based models.

## vector_store.py

Stores and retrieves embeddings using the Qdrant vector database.

## knowledge_graph.py

Builds a medical knowledge graph from extracted entities and relationships.

## retrieval.py

Implements hybrid retrieval using semantic search, keyword search, and graph-based retrieval.

## llm.py

Connects retrieved evidence with an LLM to generate final medical responses.

## multimodal.py

Handles multimodal input including dermoscopic image queries and text-image retrieval.

## evaluation.py

Evaluates retrieval performance, answer relevance, and overall system quality.

---

# Technologies Used

- Python
- Qdrant
- Sentence Transformers
- BM25
- NetworkX
- FastAPI
- Groq API
- PubMed and PMC data
- HAM10000 dataset
- ISIC dataset
- Multimodal embeddings
- Retrieval-Augmented Generation

---

# How It Works

1. Collect medical literature and dermoscopic image data.
2. Preprocess documents and create text chunks.
3. Generate embeddings for text and images.
4. Store embeddings in Qdrant.
5. Build a knowledge graph from medical concepts.
6. Retrieve relevant evidence for a user query.
7. Rerank and combine retrieved evidence.
8. Generate a grounded answer using an LLM.
9. Return the answer with supporting evidence.

---

# Example Use Case

A user can ask a skin cancer related question such as:

```text
What are the clinical signs of melanoma?
```

The system retrieves relevant medical evidence from indexed sources and generates an evidence-based response.

---

# Installation

```bash
git clone https://github.com/Urwa-Binat-Khalid/Skin-Cancer-Multimodal-RAG.git

cd Skin-Cancer-Multimodal-RAG

pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file and add your API keys:

```text
GROQ_API_KEY=your_api_key_here
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
```

---

# Run the Project

```bash
python data_ingestion.py

python document_processor.py

python embeddings.py

python vector_store.py

python retrieval.py
```

---

# Medical Disclaimer

This project is for research and educational purposes only. It is not a medical diagnosis tool and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

---

# Future Improvements

- Add complete FastAPI deployment
- Add Streamlit or Gradio interface
- Improve multimodal image reasoning
- Add more evaluation metrics
- Add Docker support
- Add CI/CD workflow
- Add sample queries and outputs

---

# License

This project is licensed under the MIT License.
