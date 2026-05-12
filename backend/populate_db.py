import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json

knowledge_base = [
    "SIP stands for Systematic Investment Plan. It is a way to invest a fixed amount regularly in mutual funds.",
    "FD stands for Fixed Deposit. It is a low-risk financial instrument provided by banks which provides investors a higher rate of interest than a regular savings account, until the given maturity date.",
    "Diversification reduces risk by allocating investments among various financial instruments, industries, and other categories.",
    "An emergency fund is a bank account with money set aside to pay for large, unexpected expenses, such as vehicle repair, medical bills, or job loss. It generally covers 3-6 months of living expenses.",
    "Mutual funds pool money from the investing public and use that money to buy other securities, usually stocks and bonds.",
    "50/30/20 rule is a budgeting guide: spend 50% on needs, 30% on wants, and stash 20% to savings."
]

def populate_db():
    VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), '../vector_db')
    INDEX_PATH = os.path.join(VECTOR_DB_DIR, 'finance_index.faiss')
    TEXTS_PATH = os.path.join(VECTOR_DB_DIR, 'texts.json')

    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding texts...")
    embeddings = model.encode(knowledge_base)
    embeddings = np.array(embeddings).astype('float32')

    print("Creating FAISS index...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    print("Saving index and texts...")
    faiss.write_index(index, INDEX_PATH)
    
    with open(TEXTS_PATH, 'w') as f:
        json.dump(knowledge_base, f)
        
    print(f"Setup complete. Stored {index.ntotal} items in FAISS memory.")

if __name__ == '__main__':
    populate_db()
