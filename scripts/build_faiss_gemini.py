import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

def load_json(file_path):
    """Loads a JSON file from a given file path."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_chunks_and_metas(data):
    """
    Creates text chunks and metadata from the loaded data.
    Each item in the data is treated as a single chunk.
    """
    chunks = []
    metas = []
    for item in data:
        # Combine the question and answer into a single text chunk
        text_chunk = f"Question: {item['question']}\nAnswer: {item['answer']}"
        chunks.append(text_chunk)
        
        # Store metadata associated with the chunk, including its original ID
        metas.append({
            'id': item['id'],
            'text': text_chunk
        })
    return chunks, metas

def build_faiss_index(chunks, metas, output_dir):
    """
    Builds a FAISS index from text chunks and saves it along with metadata.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    # Save the index and metadata
    index_path = os.path.join(output_dir, 'faiss_index_gemini.idx')
    metas_path = os.path.join(output_dir, 'metas.json')
    
    faiss.write_index(index, index_path)
    with open(metas_path, 'w', encoding='utf-8') as f:
        json.dump(metas, f, indent=2)
    
    print(f"FAISS index built and saved to {index_path}")
    print(f"Metadata saved to {metas_path}")

def main():
    """Main function to build the FAISS index."""
    processed_dir = 'data/processed'
    kb_clean_path = os.path.join(processed_dir, 'kb_clean.json')
    
    try:
        data = load_json(kb_clean_path)
    except FileNotFoundError:
        print(f"Error: {kb_clean_path} not found. Please run `prepare_kb.py` first.")
        return

    chunks, metas = create_chunks_and_metas(data)
    
    faiss_output_dir = os.path.join(processed_dir, 'faiss_gemini')
    build_faiss_index(chunks, metas, faiss_output_dir)

if __name__ == '__main__':
    main()