import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class InterviewConfig:
    """Конфигурация параметров интервью"""
    max_questions: int = 10
    question_time_limit: int = 120  # секунды
    followup_time_limit: int = 60  # секунды
    
    # Триггеры для follow-up вопросов
    pause_threshold: float = 3.0  # секунды
    min_answer_words: int = 20
    suspicious_keywords: List[str] = None
    
    # Параметры видео
    video_fps: int = 25
    video_width: int = 640
    video_height: int = 480
    
    # Параметры аудио
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    
    def __post_init__(self):
        if self.suspicious_keywords is None:
            self.suspicious_keywords = ["не знаю", "не помню", "наверное", "может быть"]

@dataclass
class ModelConfig:
    """Конфигурация моделей"""
    llm_model: str = "openai/gpt-oss-120b:cerebras"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    
    # Модель для транскрибации
    asr_model: str = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
    
    # HuggingFace токен
    hf_token: str = os.environ.get("HF_TOKEN", "")

@dataclass
class SystemConfig:
    """Общая конфигурация системы"""
    session_retention_days: int = 30
    max_file_size_mb: int = 50
    supported_cv_formats: List[str] = None
    supported_job_formats: List[str] = None
    
    # Пути
    sessions_dir: str = "sessions"
    resumes_dir: str = "resumes"
    jobs_dir: str = "jobs"
    reports_dir: str = "reports"
    
    def __post_init__(self):
        if self.supported_cv_formats is None:
            self.supported_cv_formats = ['.rtf', '.docx', '.doc', '.txt', '.pdf']
        if self.supported_job_formats is None:
            self.supported_job_formats = ['.docx', '.doc', '.txt']

# Глобальные конфигурации
interview_config = InterviewConfig()
model_config = ModelConfig()
system_config = SystemConfig()
