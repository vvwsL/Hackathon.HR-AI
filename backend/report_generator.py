import os
from openai import OpenAI
from typing import Dict, Any, List
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class ReportGenerator:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ.get("HF_TOKEN", ""),
        )
        
    def generate_report(self, cv_analysis: Dict[str, Any],
                       interview_results: List[Dict[str, Any]],
                       decision: Dict[str, Any],
                       session_folder: str) -> Dict[str, Any]:
        """Генерирует комплексный отчет по кандидату"""
        
        # 1. Анализируем данные интервью
        interview_summary = self._analyze_interview_results(interview_results)
        
        # 2. Генерируем текстовый отчет с помощью LLM
        llm_report = self._generate_llm_report(
            cv_analysis, 
            interview_summary, 
            decision
        )
        
        # 3. Создаем визуализации
        self._create_visualizations(interview_summary, session_folder)
        
        # 4. Генерируем PDF отчет
        pdf_path = self._generate_pdf_report(
            cv_analysis,
            interview_summary,
            decision,
            llm_report,
            session_folder
        )
        
        # 5. Формируем финальный JSON отчет
        final_report = {
            "candidate_name": session_folder.split("_")[-1],
            "session_id": os.path.basename(session_folder),
            "timestamp": datetime.now().isoformat(),
            "cv_analysis_summary": {
                "is_suitable": cv_analysis.get("is_suitable", False),
                "matching_score": cv_analysis.get("matching_score", 0),
                "matched_skills": cv_analysis.get("matched_skills", []),
                "missing_skills": cv_analysis.get("missing_skills", [])
            },
            "interview_summary": interview_summary,
            "final_decision": decision,
            "llm_recommendation": llm_report,
            "pdf_report": pdf_path,
            "recommendation": self._get_final_recommendation(cv_analysis, decision, llm_report)
        }
        
        # Сохраняем JSON отчет
        with open(os.path.join(session_folder, "final_report.json"), "w", encoding="utf-8") as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)
        
        return final_report
    
    def _analyze_interview_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализирует результаты интервью"""
        total_questions = len([r for r in results if not r.get("is_followup", False)])
        followup_questions = len([r for r in results if r.get("is_followup", False)])
        
        # Собираем все ответы и анализ
        all_emotions = []
        all_pauses = []
        suspicious_moments = []
        
        for idx, result in enumerate(results):
            analysis = result.get("analysis", {})
            
            # Эмоции
            emotions = analysis.get("emotions", [])
            for e in emotions:
                e["question_idx"] = idx
            all_emotions.extend(emotions)
            
            # Паузы
            pauses = analysis.get("pauses", [])
            if pauses:
                all_pauses.append({
                    "question_idx": idx,
                    "question": result.get("question", ""),
                    "pause_times": pauses
                })
            
            # Подозрительные моменты
            if analysis.get("suspicious", False):
                suspicious_moments.append({
                    "question_idx": idx,
                    "question": result.get("question", ""),
                    "speech": analysis.get("speech", ""),
                    "reason": "Обнаружены признаки неуверенности"
                })
        
        # Анализ эмоций
        emotion_counts = {}
        for e in all_emotions:
            emotion = e.get("emotion", "neutral")
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return {
            "total_questions": total_questions,
            "followup_questions": followup_questions,
            "emotion_distribution": emotion_counts,
            "total_pauses": len(all_pauses),
            "suspicious_moments": suspicious_moments,
            "average_confidence": self._calculate_confidence_score(results),
            "key_concerns": self._extract_key_concerns(results)
        }
    
    def _generate_llm_report(self, cv_analysis: Dict[str, Any],
                           interview_summary: Dict[str, Any],
                           decision: Dict[str, Any]) -> str:
        """Генерирует текстовую рекомендацию с помощью LLM"""
        
        prompt = f"""
На основе следующих данных сформулируй финальную рекомендацию по кандидату.

АНАЛИЗ РЕЗЮМЕ:
- Соответствие вакансии: {cv_analysis.get('matching_score', 0)}%
- Подтвержденные навыки: {', '.join(cv_analysis.get('matched_skills', [])[:5])}
- Недостающие навыки: {', '.join(cv_analysis.get('missing_skills', [])[:5])}

РЕЗУЛЬТАТЫ ВИДЕО-ИНТЕРВЬЮ:
- Всего вопросов: {interview_summary['total_questions']}
- Уточняющих вопросов: {interview_summary['followup_questions']}
- Средний уровень уверенности: {interview_summary['average_confidence']:.1f}/10
- Количество пауз: {interview_summary['total_pauses']}
- Подозрительных моментов: {len(interview_summary['suspicious_moments'])}

АВТОМАТИЧЕСКОЕ РЕШЕНИЕ: {decision.get('final_decision', 'UNKNOWN')}
Причина: {decision.get('reason', 'Не указана')}

Сформулируй:
1. Краткое резюме по кандидату (2-3 предложения)
2. Основные сильные стороны (если есть)
3. Основные риски и сомнения
4. Финальная рекомендация: ПРИНЯТЬ / ОТКЛОНИТЬ / ПРИГЛАСИТЬ НА ДОПОЛНИТЕЛЬНОЕ СОБЕСЕДОВАНИЕ

Ответ должен быть структурированным и профессиональным.
"""
        
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b:cerebras",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def _create_visualizations(self, interview_summary: Dict[str, Any], 
                             session_folder: str):
        """Создает графики для отчета"""
        # График распределения эмоций
        emotions = interview_summary.get("emotion_distribution", {})
        if emotions:
            plt.figure(figsize=(10, 6))
            plt.bar(emotions.keys(), emotions.values())
            plt.title("Распределение эмоций во время интервью")
            plt.xlabel("Эмоция")
            plt.ylabel("Количество")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(session_folder, "emotions_chart.png"))
            plt.close()
    
    def _generate_pdf_report(self, cv_analysis: Dict[str, Any],
                           interview_summary: Dict[str, Any],
                           decision: Dict[str, Any],
                           llm_report: str,
                           session_folder: str) -> str:
        """Генерирует PDF отчет"""
        pdf_path = os.path.join(session_folder, "report.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        
        # Стили
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
            alignment=1  # центрирование
        )
        
        story = []
        
        # Заголовок
        story.append(Paragraph("ОТЧЕТ ПО КАНДИДАТУ", title_style))
        story.append(Spacer(1, 0.5*inch))
        
        # Основная информация
        story.append(Paragraph("<b>Дата:</b> " + datetime.now().strftime("%d.%m.%Y"), styles['Normal']))
        story.append(Paragraph("<b>ID сессии:</b> " + os.path.basename(session_folder), styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Результаты анализа резюме
        story.append(Paragraph("АНАЛИЗ РЕЗЮМЕ", styles['Heading2']))
        cv_data = [
            ["Параметр", "Значение"],
            ["Соответствие вакансии", f"{cv_analysis.get('matching_score', 0)}%"],
            ["Подходит по резюме", "Да" if cv_analysis.get('is_suitable', False) else "Нет"],
            ["Подтвержденных навыков", str(len(cv_analysis.get('matched_skills', [])))],
            ["Недостающих навыков", str(len(cv_analysis.get('missing_skills', [])))]
        ]
        
        cv_table = Table(cv_data, colWidths=[3*inch, 2*inch])
        cv_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(cv_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Результаты интервью
        story.append(Paragraph("РЕЗУЛЬТАТЫ ВИДЕО-ИНТЕРВЬЮ", styles['Heading2']))
        interview_data = [
            ["Параметр", "Значение"],
            ["Всего вопросов", str(interview_summary['total_questions'])],
            ["Уточняющих вопросов", str(interview_summary['followup_questions'])],
            ["Уровень уверенности", f"{interview_summary['average_confidence']:.1f}/10"],
            ["Обнаружено пауз", str(interview_summary['total_pauses'])],
            ["Подозрительных моментов", str(len(interview_summary['suspicious_moments']))]
        ]
        
        interview_table = Table(interview_data, colWidths=[3*inch, 2*inch])
        interview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(interview_table)
        story.append(PageBreak())
        
        # Рекомендация LLM
        story.append(Paragraph("ЭКСПЕРТНАЯ РЕКОМЕНДАЦИЯ", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        
        # Разбиваем текст рекомендации на параграфы
        for paragraph in llm_report.split('\n'):
            if paragraph.strip():
                story.append(Paragraph(paragraph, styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
        
        # Финальное решение
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("ФИНАЛЬНОЕ РЕШЕНИЕ", styles['Heading2']))
        
        decision_color = colors.green if decision['final_decision'] == 'PASS' else colors.red
        decision_style = ParagraphStyle(
            'Decision',
            parent=styles['Normal'],
            fontSize=16,
            textColor=decision_color,
            alignment=1
        )
        
        story.append(Paragraph(f"<b>{decision['final_decision']}</b>", decision_style))
        story.append(Paragraph(f"Причина: {decision['reason']}", styles['Normal']))
        
        # Генерируем PDF
        doc.build(story)
        
        return pdf_path
    
    def _calculate_confidence_score(self, results: List[Dict[str, Any]]) -> float:
        """Вычисляет общий уровень уверенности кандидата"""
        scores = []
        
        for result in results:
            analysis = result.get("analysis", {})
            score = 10.0
            
            # Снижаем за паузы
            pauses = len(analysis.get("pauses", []))
            score -= min(pauses * 1.5, 4)
            
            # Снижаем за подозрительность
            if analysis.get("suspicious", False):
                score -= 2
            
            # Снижаем за негативные эмоции
            emotions = analysis.get("emotions", [])
            negative_ratio = sum(1 for e in emotions if e.get("emotion") in ["fear", "angry"]) / max(len(emotions), 1)
            score -= negative_ratio * 3
            
            scores.append(max(score, 0))
        
        return np.mean(scores) if scores else 5.0
    
    def _extract_key_concerns(self, results: List[Dict[str, Any]]) -> List[str]:
        """Извлекает ключевые проблемы из интервью"""
        concerns = []
        
        for result in results:
            question = result.get("question", "")
            analysis = result.get("analysis", {})
            
            if analysis.get("suspicious", False):
                concerns.append(f"Неуверенность при ответе на: {question[:50]}...")
            
            if len(analysis.get("pauses", [])) > 2:
                concerns.append(f"Множественные паузы при ответе на: {question[:50]}...")
        
        return concerns[:3]  # Максимум 3 главных проблемы
    
    def _get_final_recommendation(self, cv_analysis: Dict[str, Any],
                                decision: Dict[str, Any],
                                llm_report: str) -> str:
        """Формирует финальную рекомендацию"""
        if decision['final_decision'] == 'PASS' and cv_analysis.get('matching_score', 0) > 70:
            return "РЕКОМЕНДУЕТСЯ к приему на работу"
        elif decision['final_decision'] == 'FAIL':
            return "НЕ РЕКОМЕНДУЕТСЯ к приему"
        else:
            return "ТРЕБУЕТСЯ дополнительное собеседование с HR"
