from flask import Flask, request, jsonify, send_file, session, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc
import os
import uuid
import json
import logging
from datetime import datetime
from werkzeug.utils import secure_filename

# Инициализация приложения
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Инициализация расширений
db = SQLAlchemy(app)
CORS(app)

# Модели базы данных
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'candidate' or 'hr'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class InterviewSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(120), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    status = db.Column(db.String(50), nullable=False)  # 'uploaded', 'interviewing', 'completed'
    resume_path = db.Column(db.String(255))
    job_description_path = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    
    # Результаты
    overall_score = db.Column(db.Integer)
    strengths = db.Column(db.Text)
    improvements = db.Column(db.Text)
    recommendations = db.Column(db.Text)
    decision = db.Column(db.String(20))  # 'PASS' or 'FAIL'

    user = db.relationship('User', backref=db.backref('sessions', lazy=True))

# Создание таблиц
with app.app_context():
    db.create_all()
    
    # Добавление тестовых пользователей, если их нет
    if not User.query.filter_by(username='candidate').first():
        candidate = User(username='candidate', password='demo123', role='candidate')
        db.session.add(candidate)
        
    if not User.query.filter_by(username='hr').first():
        hr = User(username='hr', password='hr123', role='hr')
        db.session.add(hr)
        
    db.session.commit()

# Импорт остальных модулей
try:
    from hr_hack import analyze_cv_job_match
    from question_generator import QuestionGenerator
    from video_interview import VideoInterviewer
    from report_generator import ReportGenerator
    from engine import decision_engine
    from utils import setup_directories, cleanup_old_sessions
except ImportError as e:
    print(f"Импорт модулей не удался: {e}")
    # Заглушки для демонстрации
    def analyze_cv_job_match(*args, **kwargs):
        return {
            "is_suitable": True,
            "matching_score": 85,
            "matched_skills": ["Python", "Flask", "PostgreSQL"],
            "missing_skills": ["Docker", "Kubernetes"],
            "uncertainties": [],
            "questions_to_ask": ["Расскажите о вашем опыте работы с Python?"]
        }
    
    class QuestionGenerator:
        def generate_questions(self, *args, **kwargs):
            return [{"question": "Расскажите о себе?", "type": "general"}]
    
    class VideoInterviewer:
        def conduct_interview(self, *args, **kwargs):
            return [{"analysis": {"speech": "Демо ответ", "emotions": []}}]
    
    class ReportGenerator:
        def generate_report(self, *args, **kwargs):
            return {"status": "completed", "score": 85}
    
    def decision_engine(*args, **kwargs):
        return {"final_decision": "PASS", "reason": "Демо результат"}
    
    def setup_directories():
        os.makedirs("sessions", exist_ok=True)
        os.makedirs("uploads", exist_ok=True)
    
    def cleanup_old_sessions(days=7):
        pass

# Настройка директорий
setup_directories()

# Конфигурация
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'rtf', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Инициализация компонентов
question_generator = QuestionGenerator()
video_interviewer = VideoInterviewer()
report_generator = ReportGenerator()

# Маршруты API
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    user = User.query.filter_by(username=username, password=password).first()
    if user:
        session['user_id'] = user.id
        session['username'] = user.username
        session['role'] = user.role
        
        return jsonify({
            'success': True, 
            'user': {
                'username': user.username,
                'role': user.role
            }
        })
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})

@app.route('/api/upload-resume', methods=['POST'])
def upload_resume():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Создание сессии
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + session['username']
        session_folder = os.path.join("sessions", session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        # Сохранение файла
        filename = secure_filename(file.filename)
        filepath = os.path.join(session_folder, filename)
        file.save(filepath)
        
        # Сохранение в базе данных
        interview_session = InterviewSession(
            session_id=session_id,
            user_id=session['user_id'],
            status='uploaded',
            resume_path=filepath
        )
        db.session.add(interview_session)
        db.session.commit()
        
        # Используем дефолтное описание вакансии для демо
        job_description_path = "jobs/default_job_description.docx"
        
        # Анализ резюме
        try:
            cv_analysis = analyze_cv_job_match(filepath, job_description_path)
            
            # Сохранение анализа
            analysis_path = os.path.join(session_folder, "cv_analysis.json")
            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump(cv_analysis, f, ensure_ascii=False, indent=2)
            
            # Генерация вопросов
            questions = question_generator.generate_questions(
                cv_analysis=cv_analysis,
                job_requirements=cv_analysis.get("job_requirements", {}),
                uncertainties=cv_analysis.get("uncertainties", [])
            )
            
            # Сохранение вопросов
            questions_path = os.path.join(session_folder, "questions.json")
            with open(questions_path, "w", encoding="utf-8") as f:
                json.dump({"questions": questions}, f, ensure_ascii=False, indent=2)
            
            # Обновление сессии
            interview_session.status = 'analyzed'
            db.session.commit()
            
            return jsonify({
                'success': True,
                'analysis': cv_analysis,
                'questions': questions,
                'session_id': session_id
            })
            
        except Exception as e:
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/start-interview', methods=['POST'])
def start_interview():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'Session ID required'}), 400
    
    # Поиск сессии
    interview_session = InterviewSession.query.filter_by(session_id=session_id).first()
    if not interview_session:
        return jsonify({'error': 'Session not found'}), 404
    
    # Загрузка вопросов
    questions_path = os.path.join("sessions", session_id, "questions.json")
    if not os.path.exists(questions_path):
        return jsonify({'error': 'Questions not found'}), 404
    
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    
    questions = questions_data.get('questions', [])
    
    # Обновление сессии
    interview_session.status = 'interviewing'
    db.session.commit()
    
    return jsonify({
        'success': True,
        'questions': questions
    })

@app.route('/api/process-answer', methods=['POST'])
def process_answer():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # В реальном приложении здесь будет обработка видео
    # Для демо мы просто симулируем результат
    
    data = request.json
    session_id = data.get('session_id')
    question_index = data.get('question_index')
    
    # Симуляция обработки
    import time
    time.sleep(2)
    
    # Создание мокового результата
    mock_result = {
        "question": f"Question {question_index + 1}",
        "analysis": {
            "speech": "This is a mock transcription of the answer",
            "emotions": [
                {"time": 1.5, "emotion": "neutral"},
                {"time": 3.2, "emotion": "happy"}
            ],
            "gaze": [
                {"time": 1.5, "gaze_x": 0.1, "gaze_y": 0.2},
                {"time": 3.2, "gaze_x": 0.2, "gaze_y": 0.1}
            ],
            "pauses": [],
            "suspicious": False,
            "attention": "ok"
        }
    }
    
    # Сохранение результата
    session_folder = os.path.join("sessions", session_id)
    os.makedirs(session_folder, exist_ok=True)
    
    result_path = os.path.join(session_folder, f"answer_{question_index}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(mock_result, f, ensure_ascii=False, indent=2)
    
    return jsonify({
        'success': True,
        'result': mock_result
    })

@app.route('/api/finalize-interview', methods=['POST'])
def finalize_interview():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    session_id = data.get('session_id')
    
    # Поиск сессии
    interview_session = InterviewSession.query.filter_by(session_id=session_id).first()
    if not interview_session:
        return jsonify({'error': 'Session not found'}), 404
    
    # Симуляция обработки результатов
    import time
    time.sleep(3)
    
    # Загрузка анализа резюме
    analysis_path = os.path.join("sessions", session_id, "cv_analysis.json")
    cv_analysis = {}
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r', encoding='utf-8') as f:
            cv_analysis = json.load(f)
    
    # Сбор всех ответов (в реальном приложении здесь будет чтение файлов ответов)
    interview_results = []
    for i in range(5):  # Предполагаем 5 вопросов
        answer_path = os.path.join("sessions", session_id, f"answer_{i}.json")
        if os.path.exists(answer_path):
            with open(answer_path, 'r', encoding='utf-8') as f:
                interview_results.append(json.load(f))
    
    # Принятие решения
    decision = decision_engine(interview_results)
    
    # Генерация отчета
    session_folder = os.path.join("sessions", session_id)
    final_report = report_generator.generate_report(
        cv_analysis=cv_analysis,
        interview_results=interview_results,
        decision=decision,
        session_folder=session_folder
    )
    
    # Обновление сессии в базе данных
    interview_session.status = 'completed'
    interview_session.completed_at = datetime.utcnow()
    interview_session.overall_score = final_report.get('score', 85)
    interview_session.strengths = json.dumps(final_report.get('strengths', []))
    interview_session.improvements = json.dumps(final_report.get('improvements', []))
    interview_session.recommendations = final_report.get('recommendations', '')
    interview_session.decision = final_report.get('decision', 'PASS')
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'report': final_report
    })

@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    if 'user_id' not in session or session.get('role') != 'hr':
        return jsonify({'error': 'Not authorized'}), 403
    
    # Статистика из базы данных
    total_candidates = InterviewSession.query.count()
    passed_candidates = InterviewSession.query.filter_by(decision='PASS').count()
    
    # Расчет среднего балла
    avg_score = db.session.query(db.func.avg(InterviewSession.overall_score)).scalar() or 0
    
    # Интервью сегодня
    today = datetime.utcnow().date()
    today_interviews = InterviewSession.query.filter(
        db.func.date(InterviewSession.created_at) == today
    ).count()
    
    return jsonify({
        'total_candidates': total_candidates,
        'passed_candidates': passed_candidates,
        'average_score': round(avg_score, 1),
        'today_interviews': today_interviews
    })

@app.route('/api/dashboard/candidates', methods=['GET'])
def get_candidates():
    if 'user_id' not in session or session.get('role') != 'hr':
        return jsonify({'error': 'Not authorized'}), 403
    
    # Получение всех сессий из базы данных
    sessions = InterviewSession.query.order_by(desc(InterviewSession.created_at)).all()
    
    candidates = []
    for s in sessions:
        candidates.append({
            'name': s.user.username,
            'position': 'Developer',  # В реальном приложении это бы извлекалось из анализа резюме
            'score': s.overall_score or 0,
            'status': 'passed' if s.decision == 'PASS' else 'failed',
            'date': s.created_at.strftime('%Y-%m-%d'),
            'strengths': json.loads(s.strengths) if s.strengths else [],
            'recommendations': s.recommendations or ''
        })
    
    return jsonify({'candidates': candidates})

# Обслуживание фронтенда
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path != "" and os.path.exists("frontend/" + path):
        return send_from_directory('frontend', path)
    else:
        return send_from_directory('frontend', 'index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)