import os
from openai import OpenAI
from typing import List, Dict, Any
import json

class QuestionGenerator:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ.get("HF_TOKEN", ""),
        )
        
    def generate_questions(self, cv_analysis: Dict[str, Any], 
                         job_requirements: Dict[str, Any],
                         uncertainties: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Генерирует вопросы для видео-интервью"""
        
        # Базовые вопросы из анализа CV
        questions = []
        
        # Добавляем вопросы из анализа резюме
        if "questions_to_ask" in cv_analysis:
            for q in cv_analysis["questions_to_ask"][:5]:  # максимум 5 вопросов
                questions.append({
                    "question": q,
                    "type": "from_cv_analysis",
                    "expected_time": 60,
                    "category": "general"
                })
        
        # Генерируем дополнительные вопросы на основе неопределенностей
        for uncertainty in uncertainties[:3]:  # максимум 3 уточняющих вопроса
            prompt = f"""
Сгенерируй конкретный вопрос для видео-интервью на основе следующего сомнения:
Тема: {uncertainty.get('topic', '')}
Сомнение: {uncertainty.get('concern', '')}
Что нужно уточнить: {uncertainty.get('clarification_needed', '')}

Вопрос должен быть:
- Четким и конкретным
- Требовать развернутого ответа
- Позволять оценить реальные знания/опыт кандидата

Верни только текст вопроса, без пояснений.
"""
            
            response = self.client.chat.completions.create(
                model="openai/gpt-oss-120b:cerebras",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            question_text = response.choices[0].message.content.strip()
            questions.append({
                "question": question_text,
                "type": "clarification",
                "expected_time": 90,
                "category": uncertainty.get('topic', 'general'),
                "follow_up_triggers": self._generate_followup_triggers(uncertainty)
            })
        
        # Добавляем стандартные вопросы по компетенциям
        competency_questions = self._generate_competency_questions(job_requirements)
        questions.extend(competency_questions[:2])  # максимум 2 вопроса по компетенциям
        
        return questions
    
    def _generate_followup_triggers(self, uncertainty: Dict[str, Any]) -> List[Dict[str, str]]:
        """Генерирует триггеры для последующих вопросов"""
        triggers = []
        
        # Триггер на паузу
        triggers.append({
            "type": "pause",
            "duration": 3.0,
            "follow_up": f"Можете рассказать подробнее о {uncertainty.get('topic', 'этом')}?"
        })
        
        # Триггер на короткий ответ
        triggers.append({
            "type": "short_answer",
            "min_words": 10,
            "follow_up": "Приведите конкретный пример из вашего опыта."
        })
        
        return triggers
    
    def _generate_competency_questions(self, job_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Генерирует вопросы по ключевым компетенциям"""
        questions = []
        
        # Стандартные вопросы по must-have навыкам
        must_have_skills = job_requirements.get("must_have", [])
        
        for skill in must_have_skills[:2]:
            prompt = f"""
Сгенерируй поведенческий вопрос для оценки навыка: {skill}

Вопрос должен:
- Начинаться с "Расскажите о ситуации, когда..."
- Требовать конкретного примера из опыта
- Позволять оценить реальный уровень владения навыком

Верни только текст вопроса.
"""
            
            response = self.client.chat.completions.create(
                model="openai/gpt-oss-120b:cerebras",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            questions.append({
                "question": response.choices[0].message.content.strip(),
                "type": "behavioral",
                "expected_time": 120,
                "category": "competency",
                "skill_tested": skill
            })
        
        return questions
    
    def generate_followup_question(self, original_question: str, 
                                 candidate_answer: str,
                                 detected_issue: str) -> str:
        """Генерирует уточняющий вопрос на лету"""
        prompt = f"""
На вопрос: "{original_question}"
Кандидат ответил: "{candidate_answer}"
Обнаружена проблема: {detected_issue}

Сгенерируй короткий уточняющий вопрос, который поможет прояснить ситуацию.
Вопрос должен быть конкретным и требовать четкого ответа.

Верни только текст вопроса.
"""
        
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b:cerebras",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
