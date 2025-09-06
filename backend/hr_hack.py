import os
from openai import OpenAI
import docx
import pypandoc
import json
from typing import Dict, Any, List

# Токен должен быть установлен через переменную окружения HF_TOKEN
# os.environ["HF_TOKEN"] = "your_token_here"  # Set via environment variable

class CVAnalyzer:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ.get("HF_TOKEN", ""),
        )
        
    def read_rtf(self, file_path: str) -> str:
        """Читает RTF файл"""
        return pypandoc.convert_file(file_path, 'plain')
    
    def read_docx(self, file_path: str) -> str:
        """Читает DOCX файл"""
        doc = docx.Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    
    def analyze(self, cv_path: str, job_description_path: str) -> Dict[str, Any]:
        """Анализирует соответствие резюме вакансии"""
        # Читаем файлы
        if cv_path.endswith('.rtf'):
            cv_text = self.read_rtf(cv_path)
        elif cv_path.endswith('.docx'):
            cv_text = self.read_docx(cv_path)
        else:
            with open(cv_path, 'r', encoding='utf-8') as f:
                cv_text = f.read()
                
        job_description = self.read_docx(job_description_path)
        
        # Формируем структурированный промпт
        prompt = f"""
Проанализируй соответствие резюме вакансии и верни результат в формате JSON.

РЕЗЮМЕ КАНДИДАТА:
{cv_text}

ОПИСАНИЕ ВАКАНСИИ:
{job_description}

Верни JSON со следующей структурой:
{{
    "is_suitable": true/false,
    "matching_score": 0-100,
    "matched_skills": ["список подтвержденных навыков"],
    "missing_skills": ["список отсутствующих навыков"],
    "experience_match": {{
        "required_years": X,
        "candidate_years": Y,
        "is_sufficient": true/false
    }},
    "uncertainties": [
        {{
            "topic": "название темы",
            "concern": "описание сомнения",
            "clarification_needed": "что нужно уточнить"
        }}
    ],
    "strengths": ["сильные стороны кандидата"],
    "weaknesses": ["слабые стороны кандидата"],
    "questions_to_ask": ["список вопросов для собеседования"],
    "recommendation": "краткая рекомендация",
    "job_requirements": {{
        "must_have": ["обязательные требования"],
        "nice_to_have": ["желательные требования"]
    }}
}}
"""
        
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b:cerebras",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        # Парсим JSON из ответа
        try:
            result_text = response.choices[0].message.content
            # Извлекаем JSON из текста
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = result_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Если JSON не найден, возвращаем текст как есть
                return {
                    "is_suitable": False,
                    "analysis": result_text,
                    "error": "Failed to parse structured response"
                }
        except Exception as e:
            return {
                "is_suitable": False,
                "error": str(e),
                "raw_response": response.choices[0].message.content
            }

def analyze_cv_job_match(cv_path: str, job_description_path: str) -> Dict[str, Any]:
    """Функция для обратной совместимости"""
    analyzer = CVAnalyzer()
    return analyzer.analyze(cv_path, job_description_path)
