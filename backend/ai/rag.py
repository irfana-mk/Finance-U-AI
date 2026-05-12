import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json

# Setup
VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), '../../vector_db')
INDEX_PATH = os.path.join(VECTOR_DB_DIR, 'finance_index.faiss')
TEXTS_PATH = os.path.join(VECTOR_DB_DIR, 'texts.json')

# We use a global model instance to avoid reloading if possible, but load lazily
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def retrieve_context(query: str, top_k=2):
    if not os.path.exists(INDEX_PATH) or not os.path.exists(TEXTS_PATH):
        return "No knowledge base available."

    try:
        index = faiss.read_index(INDEX_PATH)
        with open(TEXTS_PATH, 'r') as f:
            texts = json.load(f)
            
        model = get_model()
        query_vector = model.encode([query])
        
        distances, indices = index.search(np.array(query_vector).astype('float32'), top_k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(texts):
                results.append(texts[idx])
                
        return "\n".join(results)
    except Exception as e:
        print(f"RAG retrieval error: {e}")
        return ""
