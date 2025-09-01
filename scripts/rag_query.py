import json
import os
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load configuration and data
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_index_and_metas(index_path, metas_path):
    index = faiss.read_index(index_path)
    with open(metas_path, 'r', encoding='utf-8') as f:
        metas = json.load(f)
    return index, metas

def get_topic_and_subtopic_from_query(query, topic_rules):
    """
    Finds the matching topic and subtopic for a given query based on keywords.
    """
    query_lower = query.lower()
    for rule in topic_rules:
        for keyword in rule['keywords']:
            if keyword in query_lower:
                return rule['topic'], rule['subtopic']
    return None, None

def get_relevant_chunks(query, index, metas, model, k=5):
    """
    Finds the most relevant text chunks from the knowledge base using FAISS.
    """
    query_embedding = model.encode([query])
    _, I = index.search(query_embedding, k)
    
    chunks = [metas[i] for i in I[0]]
    return chunks

def generate_rag_response(query, context, model_name="gemini-1.5-flash-latest"):
    """
    Generates a response using the Gemini model with retrieved context.
    """
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name=model_name)
    
    # Construct the final prompt with the query and context
    system_prompt = (
        "You are an expert in computer science, specifically in the domains of DBMS and OOPs. "
        "Answer the user's question based on the provided context. "
        "If the context does not contain the answer, state that you cannot answer from the given information. "
        "The context is from a knowledge base of questions and answers. Be concise and helpful."
    )
    
    full_prompt = (
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )
    
    try:
        response = model.generate_content(
            full_prompt,
            system_instruction=system_prompt
        )
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

def main(user_query):
    """Main function to perform the RAG query."""
    # Define paths
    faiss_dir = 'data/processed/faiss_gemini'
    topic_rules_path = 'config/topic_rules.json'

    # Load topic rules
    topic_rules = load_json(topic_rules_path)

    # Determine topic and subtopic from the query
    topic, subtopic = get_topic_and_subtopic_from_query(user_query, topic_rules)

    if topic:
        print(f"Detected Topic: {topic}, Subtopic: {subtopic}")
    else:
        print("No specific topic detected, performing a general search.")

    # Load the FAISS index and metadata
    try:
        index_path = os.path.join(faiss_dir, 'faiss_index_gemini.idx')
        metas_path = os.path.join(faiss_dir, 'metas.json')
        index, metas = load_index_and_metas(index_path, metas_path)
    except FileNotFoundError:
        return "Error: FAISS index files not found. Please run `build_faiss_gemini.py` first."

    # Initialize the Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Add topic/subtopic information to the query for better search results
    if topic and subtopic:
        augmented_query = f"Question about {subtopic} in {topic}: {user_query}"
    else:
        augmented_query = user_query

    # Get relevant chunks
    relevant_chunks = get_relevant_chunks(augmented_query, index, metas, model)
    context_text = "\n\n".join([chunk['text'] for chunk in relevant_chunks])

    # Generate the response
    response_text = generate_rag_response(user_query, context_text)

    # Print the result
    print("\n--- RAG Response ---")
    print(response_text)
    
    print("\n--- Source Chunks (for debugging) ---")
    for chunk in relevant_chunks:
        print(f"Source ID: {chunk['id']}")
        print(f"Text: {chunk['text']}\n")

if __name__ == '__main__':
    user_question = input("Enter your question: ")
    main(user_question)