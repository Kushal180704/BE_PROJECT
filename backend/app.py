from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import PyPDF2
import docx
from io import BytesIO
import json
import random

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///interview_prep.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('backend/instance', exist_ok=True)

db = SQLAlchemy(app)
CORS(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

# Template filters
@app.template_filter('from_json')
def from_json_filter(json_str):
    if not json_str:
        return []
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return []

@app.template_filter('datetime_diff')
def datetime_diff_filter(dt):
    if not dt:
        return datetime.now() - datetime.now()
    return datetime.now() - dt

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    full_name = db.Column(db.String(100), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    experience_years = db.Column(db.Integer, default=0)
    skills = db.Column(db.Text, nullable=True)
    resume_filename = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    interviews = db.relationship('InterviewSession', backref='user', lazy=True)

class InterviewSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_type = db.Column(db.String(50), nullable=False)
    questions = db.Column(db.Text, nullable=True)
    score = db.Column(db.Float, nullable=True)
    feedback = db.Column(db.Text, nullable=True)
    duration = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize RAG components
model = None
index = None
metas = None
topic_rules = None

def initialize_rag_system():
    global model, index, metas, topic_rules
    try:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ Warning: No GEMINI_API_KEY found in environment.")
            return False
        
        genai.configure(api_key=api_key)
        
        try:
            test_model = genai.GenerativeModel("models/gemini-2.5-flash")
            test_response = test_model.generate_content("Hello")
            print("✅ Gemini API connection successful!")
        except Exception as gemini_error:
            print(f"⚠️ Gemini API error: {gemini_error}")
            return False
        
        try:
            topic_rules_path = 'config/topic_rules.json'
            with open(topic_rules_path, 'r', encoding='utf-8') as f:
                topic_rules = json.load(f)
            
            faiss_dir = 'data/processed/faiss_gemini'
            index_path = os.path.join(faiss_dir, 'faiss_index_gemini.idx')
            metas_path = os.path.join(faiss_dir, 'metas.json')
            
            index = faiss.read_index(index_path)
            with open(metas_path, 'r', encoding='utf-8') as f:
                metas = json.load(f)
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ RAG system initialized successfully!")
        except Exception as rag_error:
            print(f"⚠️ RAG system files not found: {rag_error}")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return False

# Utility functions
def get_topic_and_subtopic_from_query(query, topic_rules):
    if not topic_rules:
        return None, None
    query_lower = query.lower()
    for rule in topic_rules:
        for keyword in rule['keywords']:
            if keyword in query_lower:
                return rule['topic'], rule['subtopic']
    return None, None

def get_relevant_chunks(query, index, metas, model, k=5):
    if not all([index, metas, model]):
        return []
    query_embedding = model.encode([query])
    _, I = index.search(query_embedding, k)
    chunks = [metas[i] for i in I[0]]
    return chunks

def generate_rag_response(query, context):
    try:
        gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
        full_prompt = (
            f"You are an expert computer science interview assistant specializing in DBMS, OOPs, and Operating Systems.\n"
            f"Provide a clear, concise answer based on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer in a structured way that would help in interview preparation:"
        )
        response = gemini_model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def extract_text_from_pdf(file_stream):
    try:
        pdf_reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return None

def extract_text_from_docx(file_stream):
    try:
        doc = docx.Document(file_stream)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting DOCX text: {e}")
        return None

def parse_resume_text(text):
    lines = text.strip().split('\n')
    tech_keywords = [
        'python', 'java', 'javascript', 'react', 'node', 'sql', 'html', 'css', 
        'machine learning', 'data science', 'flask', 'django', 'mongodb', 'mysql'
    ]
    skills = []
    for line in lines:
        line_lower = line.lower()
        for keyword in tech_keywords:
            if keyword in line_lower and keyword not in [s.lower() for s in skills]:
                skills.append(keyword.title())
    experience_years = 0
    for line in lines:
        if 'year' in line.lower() and 'experience' in line.lower():
            words = line.split()
            for i, word in enumerate(words):
                if word.isdigit() and i < len(words) - 1 and 'year' in words[i+1].lower():
                    experience_years = int(word)
                    break
    return {
        'skills': skills,
        'experience_years': experience_years,
        'raw_text': text[:1000]
    }

# ============ ROUTES ============

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': 'Invalid username or password'})
    return render_template('auth/login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        full_name = data.get('full_name')
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Username already exists'})
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'message': 'Email already registered'})
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            full_name=full_name
        )
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return jsonify({'success': True, 'message': 'Registration successful'})
    return render_template('auth/signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    data = request.get_json()
    current_user.full_name = data.get('full_name', current_user.full_name)
    current_user.phone = data.get('phone', current_user.phone)
    current_user.experience_years = data.get('experience_years', current_user.experience_years)
    current_user.skills = json.dumps(data.get('skills', []))
    db.session.commit()
    return jsonify({'success': True, 'message': 'Profile updated successfully'})

@app.route('/upload_resume', methods=['GET', 'POST'])
@login_required
def upload_resume():
    if request.method == 'POST':
        if 'resume' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'})
        file = request.files['resume']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        if file:
            filename = secure_filename(f"{current_user.id}_{file.filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_stream = BytesIO()
            file.stream.seek(0)
            file_stream.write(file.stream.read())
            file_stream.seek(0)
            text = None
            if filename.lower().endswith('.pdf'):
                text = extract_text_from_pdf(file_stream)
            elif filename.lower().endswith('.docx'):
                text = extract_text_from_docx(file_stream)
            if text:
                resume_data = parse_resume_text(text)
                current_user.resume_filename = filename
                current_user.skills = json.dumps(resume_data['skills'])
                current_user.experience_years = resume_data['experience_years']
                db.session.commit()
                return jsonify({
                    'success': True, 
                    'message': 'Resume uploaded and parsed successfully',
                    'data': resume_data
                })
            else:
                return jsonify({'success': False, 'message': 'Could not extract text from resume'})
    return render_template('resume_upload.html')

@app.route('/technical_chat')
@login_required
def technical_chat():
    return render_template('rag_chat.html')

@app.route('/hr_interview')
@login_required
def hr_interview():
    return render_template('hr_interview.html')

@app.route('/api/query', methods=['POST'])
@login_required
def rag_query():
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        if not user_query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        if not all([index, metas, model, topic_rules]):
            return jsonify({'error': 'RAG system not initialized. Please contact administrator.'}), 500
        
        topic, subtopic = get_topic_and_subtopic_from_query(user_query, topic_rules)
        if topic and subtopic:
            augmented_query = f"Question about {subtopic} in {topic}: {user_query}"
        else:
            augmented_query = user_query
        relevant_chunks = get_relevant_chunks(augmented_query, index, metas, model)
        context_text = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
        response_text = generate_rag_response(user_query, context_text)
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

# ============ ENHANCED HR QUESTIONS ROUTE ============
@app.route('/api/hr_questions', methods=['POST'])
@login_required
def generate_hr_questions():
    """Generate VARIED HR interview questions using randomization"""
    try:
        skills = json.loads(current_user.skills) if current_user.skills else []
        experience = current_user.experience_years
        random_seed = random.randint(1, 10000)
        
        gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
        
        prompt = f"""
        Generate 5 UNIQUE and VARIED HR interview questions for a candidate with:
        - Skills: {', '.join(skills) if skills else 'General'}
        - Experience: {experience} years
        
        MUST DO:
        1. Generate COMPLETELY DIFFERENT questions each time (use randomization seed: {random_seed})
        2. Mix types: behavioral, situational, technical, career, leadership
        3. Include teamwork and conflict questions
        4. Include skill-specific questions
        5. Vary difficulty based on experience
        
        Return VALID JSON ONLY (no markdown):
        [
            {{"question": "...?", "type": "behavioral", "focus_area": "teamwork"}},
            {{"question": "...?", "type": "situational", "focus_area": "problem_solving"}},
            {{"question": "...?", "type": "career", "focus_area": "growth"}},
            {{"question": "...?", "type": "leadership", "focus_area": "influence"}},
            {{"question": "...?", "type": "technical", "focus_area": "skills"}}
        ]
        """
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
        
        try:
            questions = json.loads(response_text)
            if not isinstance(questions, list) or len(questions) == 0:
                raise ValueError("Invalid format")
        except:
            fallback_questions = [
                {"question": "Tell me about yourself", "type": "general", "focus_area": "introduction"},
                {"question": "Why are you interested in this role?", "type": "situational", "focus_area": "motivation"},
                {"question": "Describe a challenge you overcame", "type": "behavioral", "focus_area": "problem_solving"},
                {"question": "How do you handle conflict with team members?", "type": "behavioral", "focus_area": "teamwork"},
                {"question": "Where do you see yourself in 5 years?", "type": "career", "focus_area": "growth"}
            ]
            questions = random.sample(fallback_questions, 5)
        
        return jsonify({'success': True, 'questions': questions})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============ REAL-TIME AI FEEDBACK ROUTE ============
@app.route('/api/evaluate_answer', methods=['POST'])
@login_required
def evaluate_answer():
    """Real-time AI evaluation of HR answers with detailed feedback"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        answer = data.get('answer', '').strip()
        question_type = data.get('type', 'general')
        focus_area = data.get('focus_area', 'general')
        
        if not answer or len(answer) < 10:
            return jsonify({
                'success': False,
                'error': 'Please provide a more detailed answer'
            }), 400
        
        gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
        
        evaluation_prompt = f"""
        You are an expert HR interviewer. Analyze this interview response DEEPLY and provide comprehensive feedback.
        
        QUESTION: {question}
        QUESTION TYPE: {question_type}
        FOCUS AREA: {focus_area}
        
        CANDIDATE ANSWER: "{answer}"
        
        Provide evaluation in VALID JSON format ONLY (no markdown, no code blocks):
        {{
            "score": <1-10>,
            "overall_feedback": "2-3 sentence summary",
            "strengths": ["strength 1", "strength 2", "strength 3"],
            "improvements": ["improvement 1", "improvement 2", "improvement 3"],
            "suggestions": [
                {{"category": "Structure", "tip": "specific tip"}},
                {{"category": "Content", "tip": "specific tip"}},
                {{"category": "Communication", "tip": "specific tip"}}
            ],
            "star_method": {{
                "situation": {{"present": true/false, "feedback": "comment"}},
                "task": {{"present": true/false, "feedback": "comment"}},
                "action": {{"present": true/false, "feedback": "comment"}},
                "result": {{"present": true/false, "feedback": "comment"}}
            }},
            "key_metrics": {{
                "clarity": <1-10>,
                "relevance": <1-10>,
                "confidence": <1-10>,
                "specificity": <1-10>
            }},
            "next_steps": ["action 1", "action 2", "action 3"]
        }}
        
        Evaluate on: clarity, relevance, STAR method, examples, confidence, professionalism.
        """
        
        response = gemini_model.generate_content(evaluation_prompt)
        response_text = response.text.strip()
        
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
        
        try:
            feedback = json.loads(response_text)
        except:
            feedback = {
                "score": 6,
                "overall_feedback": "Good attempt. Add more specific examples and structure your answer using STAR method.",
                "strengths": ["You addressed the question", "Clear communication"],
                "improvements": ["Add more concrete examples", "Structure better with STAR method"],
                "suggestions": [
                    {"category": "Structure", "tip": "Use STAR method: Situation, Task, Action, Result"},
                    {"category": "Content", "tip": "Add measurable outcomes"},
                    {"category": "Communication", "tip": "Be more confident and detailed"}
                ],
                "star_method": {
                    "situation": {"present": False, "feedback": "Set context"},
                    "task": {"present": False, "feedback": "Explain your role"},
                    "action": {"present": True, "feedback": "Good action description"},
                    "result": {"present": False, "feedback": "What was the outcome?"}
                },
                "key_metrics": {
                    "clarity": 6,
                    "relevance": 6,
                    "confidence": 5,
                    "specificity": 5
                },
                "next_steps": ["Practice STAR method", "Add more examples", "Rehearse your answer"]
            }
        
        return jsonify({
            'success': True,
            'feedback': feedback
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============ EXISTING ROUTES (unchanged) ============

@app.route('/api/user_stats', methods=['GET'])
@login_required
def get_user_stats():
    try:
        sessions_count = InterviewSession.query.filter_by(user_id=current_user.id).count()
        completed_sessions = InterviewSession.query.filter(
            InterviewSession.user_id == current_user.id,
            InterviewSession.score.isnot(None)
        ).all()
        avg_score = None
        if completed_sessions:
            avg_score = sum(session.score for session in completed_sessions) / len(completed_sessions)
        questions_answered = sessions_count * 5
        return jsonify({
            'sessions': sessions_count,
            'questions': questions_answered,
            'avg_score': avg_score
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/topics', methods=['GET'])
def get_topics():
    try:
        with open('config/taxonomy.json', 'r', encoding='utf-8') as f:
            taxonomy = json.load(f)
        return jsonify(taxonomy)
    except Exception as e:
        return jsonify({'error': f'Error loading topics: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'rag_initialized': all([model is not None, index is not None, metas is not None, topic_rules is not None])
    })

@app.route('/api/save_interview_session', methods=['POST'])
@login_required
def save_interview_session():
    try:
        data = request.get_json()
        session_type = data.get('session_type', 'hr')
        questions = data.get('questions', [])
        answers = data.get('answers', {})
        
        session = InterviewSession(
            user_id=current_user.id,
            session_type=session_type,
            questions=json.dumps({
                'questions': questions,
                'answers': answers
            }),
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
        )
        
        db.session.add(session)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Session saved successfully',
            'session_id': session.id
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/interview_history', methods=['GET'])
@login_required
def get_interview_history():
    try:
        session_type = request.args.get('type', None)
        query = InterviewSession.query.filter_by(user_id=current_user.id)
        
        if session_type:
            query = query.filter_by(session_type=session_type)
        
        sessions = query.order_by(InterviewSession.created_at.desc()).limit(20).all()
        
        history = []
        for session in sessions:
            questions_data = json.loads(session.questions) if session.questions else {}
            history.append({
                'id': session.id,
                'session_type': session.session_type,
                'created_at': session.created_at.isoformat(),
                'completed_at': session.completed_at.isoformat() if session.completed_at else None,
                'questions_count': len(questions_data.get('questions', [])),
                'score': session.score,
                'feedback': session.feedback
            })
        
        return jsonify({
            'success': True,
            'sessions': history
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    initialize_rag_system()
    print("🚀 Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)