# BE PROJECT
# Knowledge Base RAG System

A comprehensive Question-Answering system using Retrieval-Augmented Generation (RAG) for Computer Science topics including DBMS, Object-Oriented Programming (OOPs), and Operating Systems (OS).

## 📁 Project Structure

```
BEPROJECT/
├── config/
│   ├── taxonomy.json          # Topic hierarchy and subtopics
│   └── topic_rules.json       # Keyword mappings for topic classification
├── data/
│   ├── raw/                   # Raw knowledge base files
│   │   ├── complete_dbms.json
│   │   ├── oops_qna_simplified.json
│   │   └── os_qna.json        # NEW: OS questions (100 questions)
│   └── processed/             # Processed data and embeddings
│       ├── kb_clean.json      # Clean, categorized Q&A data
│       ├── kb_chunks.jsonl    # Text chunks for embedding
│       ├── kg_edges.csv       # Knowledge graph relationships
│       └── faiss_gemini/      # FAISS vector index
│           ├── faiss_index_gemini.idx
│           ├── metas.json
│           └── ids.json
└── scripts/
    ├── prepare_kb.py          # Data preprocessing and topic assignment
    ├── build_faiss_gemini.py  # FAISS index construction
    └── rag_query.py           # Query interface for RAG system
```

## 🚀 Features

### Knowledge Domains
- **DBMS**: Database Management Systems (15 subtopics)
- **OOPs**: Object-Oriented Programming (8 subtopics) 
- **OS**: Operating Systems (10 subtopics) - **NEWLY ADDED**

### OS Subtopics Include:
- General OS Concepts
- Process Management
- Memory Management
- Synchronization
- File Systems
- I/O Management
- System Calls
- Networking
- Storage Systems
- Security

### System Capabilities
- Automatic topic and subtopic classification
- Difficulty level assignment (Beginner/Intermediate/Advanced)
- Vector similarity search using FAISS
- Context-aware response generation with Gemini AI
- Knowledge graph construction from taxonomy

## 🛠️ Setup and Installation

### Prerequisites
```bash
pip install -r requirements.txt
```

Required packages:
- `sentence-transformers`
- `faiss-cpu` (or `faiss-gpu` for GPU acceleration)
- `google-generativeai`
- `unidecode`
- `numpy`
- `pathlib`

### Environment Setup
Set your Gemini API key:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

## 📋 Usage Instructions

### Step 1: Prepare the Knowledge Base
Process raw JSON files and create clean, categorized data:

```bash
cd scripts/
python prepare_kb.py
```

This will:
- Load all JSON files from `data/raw/`
- Normalize and clean the text
- Assign topics and subtopics using keyword rules
- Determine difficulty levels
- Generate processed files in `data/processed/`

### Step 2: Build FAISS Index
Create vector embeddings and search index:

```bash
python build_faiss_gemini.py
```

This will:
- Generate embeddings for all Q&A pairs
- Create FAISS index for fast similarity search
- Save index and metadata files

### Step 3: Query the System
Run interactive queries:

```bash
python rag_query.py
```

Example queries:
- "What is deadlock in operating systems?"
- "Explain virtual memory and paging"
- "What are the ACID properties in DBMS?"
- "What is polymorphism in OOP?"

## 📊 Data Statistics

- **Total Questions**: ~300+ across all domains
- **DBMS**: ~185 questions with 15 subtopics
- **OOPs**: ~200 questions with 8 subtopics  
- **OS**: 100 questions with 10 subtopics
- **Embedding Model**: all-MiniLM-L6-v2
- **Response Generation**: Gemini 1.5 Flash

## 🔧 Configuration

### Adding New Topics
1. Update `config/taxonomy.json` with new topic structure
2. Add keyword rules in `config/topic_rules.json`
3. Place raw JSON data in `data/raw/`
4. Re-run preparation and indexing steps

### Keyword Rule Format
```json
{
  "keywords": ["process", "thread", "scheduling"],
  "topic": "OS", 
  "subtopic": "Process Management"
}
```

### Raw Data Format
```json
[
  {
    "id": 1,
    "question": "What is an operating system?",
    "answer": "An operating system is software that manages..."
  }
]
```

## 🎯 System Workflow

1. **Data Ingestion**: Raw JSON files → Normalized text
2. **Topic Classification**: Keyword matching → Topic/subtopic assignment  
3. **Difficulty Assessment**: Heuristic analysis → Beginner/Intermediate/Advanced
4. **Vectorization**: Text → Embeddings using SentenceTransformers
5. **Indexing**: Embeddings → FAISS index for fast retrieval
6. **Query Processing**: User query → Topic detection → Vector search
7. **Response Generation**: Retrieved context + Query → Gemini AI → Answer

## 🔍 Advanced Features

### Topic-Aware Search
The system automatically detects query topics and enhances search:
```python
# Query: "What is process scheduling?"
# Detected: Topic=OS, Subtopic=Process Management  
# Enhanced query: "Question about Process Management in OS: What is process scheduling?"
```

### Difficulty-Based Filtering
Questions are automatically categorized by difficulty based on content analysis.

### Knowledge Graph Integration
Taxonomy relationships are exported as CSV for graph analysis and visualization.

## 🚨 Troubleshooting

### Common Issues
1. **"FAISS index not found"**: Run `build_faiss_gemini.py` first
2. **"kb_clean.json not found"**: Run `prepare_kb.py` first  
3. **Gemini API errors**: Check your API key environment variable
4. **Topic not detected**: Add relevant keywords to `topic_rules.json`

### Performance Tips
- Use GPU-enabled FAISS for larger datasets
- Adjust embedding batch size for memory constraints
- Tune the number of retrieved chunks (k parameter)

## 📈 Recent Updates

### Version 2.0 - OS Knowledge Base Added
- ✅ Added 100 Operating Systems questions
- ✅ Created 10 OS subtopics with keyword mappings
- ✅ Enhanced difficulty detection for OS concepts
- ✅ Updated system prompt to include OS expertise
- ✅ Maintained backward compatibility with DBMS and OOPs

## 🤝 Contributing

To add new knowledge domains:
1. Prepare JSON file with question-answer pairs
2. Define subtopics and keywords in configuration
3. Update taxonomy structure
4. Test topic classification accuracy
5. Rebuild index and validate results

## 📜 License

This project is for educational purposes. Ensure proper attribution when using the knowledge base content.
