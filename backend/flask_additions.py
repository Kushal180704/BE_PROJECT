# Add these template filters to your app.py after the app initialization

import json
from datetime import datetime

@app.template_filter('from_json')
def from_json_filter(json_str):
    """Convert JSON string to Python object"""
    if not json_str:
        return []
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return []

@app.template_filter('datetime_diff')
def datetime_diff_filter(dt):
    """Calculate difference between datetime and now"""
    if not dt:
        return datetime.now() - datetime.now()
    return datetime.now() - dt

# Add user stats API endpoint
@app.route('/api/user_stats', methods=['GET'])
@login_required
def get_user_stats():
    """Get user statistics for dashboard"""
    try:
        sessions_count = InterviewSession.query.filter_by(user_id=current_user.id).count()
        
        # Calculate average score
        completed_sessions = InterviewSession.query.filter(
            InterviewSession.user_id == current_user.id,
            InterviewSession.score.isnot(None)
        ).all()
        
        avg_score = None
        if completed_sessions:
            avg_score = sum(session.score for session in completed_sessions) / len(completed_sessions)
        
        # Count questions answered (mock data for now)
        questions_answered = sessions_count * 5  # Assume 5 questions per session
        
        return jsonify({
            'sessions': sessions_count,
            'questions': questions_answered,
            'avg_score': avg_score
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500