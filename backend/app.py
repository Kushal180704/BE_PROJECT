from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize global variables
model = None
index = None
metas = None
topic_rules = None

def initialize_rag_system():
    """Initialize the RAG system components"""
    global model, index, metas, topic_rules
    
    try:
        # Configure Gemini API
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing API key: set GEMINI_API_KEY or GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        
        # Load topic rules
        topic_rules_path = 'config/topic_rules.json'
        with open(topic_rules_path, 'r', encoding='utf-8') as f:
            topic_rules = json.load(f)
        
        # Load FAISS index and metadata
        faiss_dir = 'data/processed/faiss_gemini'
        index_path = os.path.join(faiss_dir, 'faiss_index_gemini.idx')
        metas_path = os.path.join(faiss_dir, 'metas.json')
        
        index = faiss.read_index(index_path)
        with open(metas_path, 'r', encoding='utf-8') as f:
            metas = json.load(f)
        
        # Initialize sentence transformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("✅ RAG system initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return False

def get_topic_and_subtopic_from_query(query, topic_rules):
    """Find matching topic and subtopic for query"""
    query_lower = query.lower()
    for rule in topic_rules:
        for keyword in rule['keywords']:
            if keyword in query_lower:
                return rule['topic'], rule['subtopic']
    return None, None

def get_relevant_chunks(query, index, metas, model, k=5):
    """Find most relevant chunks using FAISS"""
    query_embedding = model.encode([query])
    _, I = index.search(query_embedding, k)
    chunks = [metas[i] for i in I[0]]
    return chunks

def generate_rag_response(query, context, model_name="gemini-1.5-flash-latest"):
    """Generate response using Gemini"""
    try:
        model = genai.GenerativeModel(model_name=model_name)
        
        full_prompt = (
            f"You are an expert computer science interview assistant specializing in DBMS, OOPs, and Operating Systems.\n"
            f"Provide a clear, concise answer based on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer in a structured way that would help in interview preparation:"
        )
        
        response = model.generate_content(full_prompt)
        return response.text
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.route('/')
def home():
    """Serve the main frontend page"""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query():
    """Handle RAG queries from frontend"""
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        
        if not user_query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Detect topic and subtopic
        topic, subtopic = get_topic_and_subtopic_from_query(user_query, topic_rules)
        
        # Augment query for better search
        if topic and subtopic:
            augmented_query = f"Question about {subtopic} in {topic}: {user_query}"
        else:
            augmented_query = user_query
        
        # Get relevant chunks
        relevant_chunks = get_relevant_chunks(augmented_query, index, metas, model)
        context_text = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
        
        # Generate response
        response_text = generate_rag_response(user_query, context_text)
        
        # Prepare response data
        response_data = {
            'success': True,
            'query': user_query,
            'answer': response_text,
            'detected_topic': topic,
            'detected_subtopic': subtopic,
            'source_count': len(relevant_chunks),
            'sources': [{'id': chunk['id'], 'text': chunk['text'][:200] + '...'} for chunk in relevant_chunks[:3]]
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/topics', methods=['GET'])
def get_topics():
    """Get available topics and subtopics"""
    try:
        # Load taxonomy
        with open('config/taxonomy.json', 'r', encoding='utf-8') as f:
            taxonomy = json.load(f)
        return jsonify(taxonomy)
    except Exception as e:
        return jsonify({'error': f'Error loading topics: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'rag_initialized': all([model is not None, index is not None, metas is not None, topic_rules is not None])
    })

if __name__ == '__main__':
    # Initialize RAG system on startup
    if initialize_rag_system():
        print("🚀 Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to start server due to initialization error")
