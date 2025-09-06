import os
import json
import time
import threading
from typing import List, Dict, Any
import cv2
import numpy as np
from datetime import datetime

from interview_analysis import run_interview
from question_generator import QuestionGenerator

class VideoInterviewer:
    def __init__(self):
        self.question_generator = QuestionGenerator()
        self.current_session = None
        
    def conduct_interview(self, questions: List[Dict[str, Any]], 
                         session_folder: str) -> List[Dict[str, Any]]:
        """Проводит видео-интервью с кандидатом"""
        self.current_session = session_folder
        all_results = []
        
        print("\n" + "="*60)
        print("НАЧАЛО ВИДЕО-ИНТЕРВЬЮ")
        print("="*60)
        
        for idx, question_data in enumerate(questions):
            print(f"\nВопрос {idx + 1} из {len(questions)}")
            print("-" * 40)
            
            # Основной вопрос
            result = self._ask_question(
                question_data["question"],
                question_data.get("expected_time", 60),
                f"question_{idx + 1}"
            )
            
            result["question_data"] = question_data
            all_results.append(result)
            
            # Анализируем необходимость follow-up вопросов
            if self._needs_followup(result, question_data):
                followup_q = self._generate_dynamic_followup(
                    question_data["question"],
                    result
                )
                
                print(f"\nУточняющий вопрос:")
                followup_result = self._ask_question(
                    followup_q,
                    30,
                    f"question_{idx + 1}_followup"
                )
                
                followup_result["is_followup"] = True
                followup_result["parent_question"] = idx
                all_results.append(followup_result)
            
            # Пауза между вопросами
            time.sleep(2)
        
        print("\n" + "="*60)
        print("ИНТЕРВЬЮ ЗАВЕРШЕНО")
        print("="*60)
        
        return all_results
    
    def _ask_question(self, question_text: str, duration: int, 
                     question_id: str) -> Dict[str, Any]:
        """Задает вопрос и записывает ответ"""
        print(f"\nВопрос: {question_text}")
        print(f"У вас есть {duration} секунд на ответ.")
        print("Нажмите Enter, когда будете готовы начать...")
        input()
        
        # Создаем папку для этого вопроса
        question_folder = os.path.join(self.current_session, question_id)
        os.makedirs(question_folder, exist_ok=True)
        
        # Запускаем анализ
        result = run_interview(question_text)
        
        # Перемещаем файлы в нужную папку
        if "analysis" in result and "video" in result["analysis"]:
            video_path = result["analysis"]["video"]
            if os.path.exists(video_path):
                new_path = os.path.join(question_folder, os.path.basename(video_path))
                os.rename(video_path, new_path)
                result["analysis"]["video"] = new_path
        
        return result
    
    def _needs_followup(self, result: Dict[str, Any], 
                       question_data: Dict[str, Any]) -> bool:
        """Определяет необходимость уточняющего вопроса"""
        analysis = result.get("analysis", {})
        
        # Проверяем триггеры
        if analysis.get("suspicious", False):
            return True
            
        # Проверяем длительные паузы
        if len(analysis.get("pauses", [])) > 1:
            return True
            
        # Проверяем короткий ответ
        speech = analysis.get("speech", "")
        if len(speech.split()) < 20 and question_data.get("type") != "clarification":
            return True
            
        # Проверяем эмоциональное состояние
        emotions = analysis.get("emotions", [])
        negative_emotions = sum(1 for e in emotions if e.get("emotion") in ["fear", "angry", "sad"])
        if negative_emotions > len(emotions) * 0.3:
            return True
            
        return False
    
    def _generate_dynamic_followup(self, original_question: str, 
                                  result: Dict[str, Any]) -> str:
        """Генерирует уточняющий вопрос на основе анализа"""
        analysis = result.get("analysis", {})
        speech = analysis.get("speech", "")
        
        # Определяем причину для follow-up
        if analysis.get("suspicious", False):
            issue = "заминки и неуверенность в ответе"
        elif len(analysis.get("pauses", [])) > 1:
            issue = "длительные паузы в речи"
        elif len(speech.split()) < 20:
            issue = "слишком короткий ответ"
        else:
            issue = "эмоциональное напряжение"
        
        return self.question_generator.generate_followup_question(
            original_question,
            speech,
            issue
        )
