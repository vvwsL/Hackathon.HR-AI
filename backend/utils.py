import os
import shutil
import json
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def setup_directories():
    """Создает необходимые директории"""
    directories = [
        "sessions",
        "resumes", 
        "jobs",
        "reports",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Создана директория: {directory}")

def cleanup_old_sessions(days: int = 7):
    """Удаляет старые сессии"""
    import time
    from datetime import datetime, timedelta
    
    sessions_dir = "sessions"
    if not os.path.exists(sessions_dir):
        return
    
    cutoff_time = time.time() - (days * 24 * 60 * 60)
    
    for session in os.listdir(sessions_dir):
        session_path = os.path.join(sessions_dir, session)
        if os.path.isdir(session_path):
            if os.path.getmtime(session_path) < cutoff_time:
                shutil.rmtree(session_path)
                logger.info(f"Удалена старая сессия: {session}")

def validate_file_formats(cv_path: str, job_path: str) -> bool:
    """Проверяет форматы файлов"""
    valid_cv_formats = ['.rtf', '.docx', '.doc', '.txt', '.pdf']
    valid_job_formats = ['.docx', '.doc', '.txt']
    
    cv_ext = os.path.splitext(cv_path)[1].lower()
    job_ext = os.path.splitext(job_path)[1].lower()
    
    if cv_ext not in valid_cv_formats:
        logger.error(f"Неподдерживаемый формат резюме: {cv_ext}")
        return False
    
    if job_ext not in valid_job_formats:
        logger.error(f"Неподдерживаемый формат описания вакансии: {job_ext}")
        return False
    
    return True

def merge_interview_videos(session_folder: str) -> str:
    """Объединяет все видео интервью в один файл"""
    try:
        import cv2
        
        video_files = []
        for root, dirs, files in os.walk(session_folder):
            for file in files:
                if file.endswith('.avi'):
                    video_files.append(os.path.join(root, file))
        
        if not video_files:
            return None
        
        # Сортируем по времени создания
        video_files.sort(key=os.path.getctime)
        
        # Читаем параметры первого видео
        cap = cv2.VideoCapture(video_files[0])
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Создаем выходной файл
        output_path = os.path.join(session_folder, "full_interview.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Объединяем видео
        for video_file in video_files:
            cap = cv2.VideoCapture(video_file)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            cap.release()
        
        out.release()
        logger.info(f"Создано объединенное видео: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Ошибка при объединении видео: {str(e)}")
        return None

def export_session_data(session_id: str, export_format: str = 'json') -> str:
    """Экспортирует данные сессии в указанном формате"""
    session_folder = os.path.join("sessions", session_id)
    
    if not os.path.exists(session_folder):
        raise ValueError(f"Сессия {session_id} не найдена")
    
    # Собираем все данные
    session_data = {
        "session_id": session_id,
        "files": {}
    }
    
    # Читаем все JSON файлы
    for file in os.listdir(session_folder):
        if file.endswith('.json'):
            file_path = os.path.join(session_folder, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                session_data["files"][file] = json.load(f)
    
    # Экспортируем
    export_path = os.path.join("reports", f"{session_id}_export.{export_format}")
    
    if export_format == 'json':
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Данные сессии экспортированы: {export_path}")
    return export_path
