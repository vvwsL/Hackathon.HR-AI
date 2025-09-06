import cv2
import os
import time
import json
import math
import numpy as np
from datetime import datetime
from collections import deque

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

# Детекция эмоций с DeepFace
try:
    from deepface import DeepFace
    EMOTIONS_ON = True
    print("✓ Эмоции включены (DeepFace)")
except ImportError:
    EMOTIONS_ON = False
    print("⚠ Эмоции отключены. Установите: pip install deepface")

# ========== НАСТРОЙКИ ==========
EMOTION_LIST = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
LMS = {"nose_tip": 1, "chin": 152, "l_eye_outer": 33, "r_eye_outer": 263, "l_mouth": 61, "r_mouth": 291}
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263]
MODEL_3D = np.array([
    (0,0,0), (0,-63.6,-12.5), (-43.3,32.7,-26), 
    (43.3,32.7,-26), (-28.9,-28.9,-24.1), (28.9,-28.9,-24.1)
], dtype=np.float64)

def create_folder():
    folder = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(folder, exist_ok=True)
    return folder

class Analyzer:
    def __init__(self):
        self.mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
        self.K = None
        self.prev_yaw = 0
        self.prev_pitch = 0
        self.gaze_hist = deque(maxlen=30)
        self.emotions = {e: 0 for e in EMOTION_LIST}
        self.last_emotion = "neutral"

    def analyze(self, frame, t):
        h, w = frame.shape[:2]
        if self.K is None:
            self.K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)

        result = {
            "time": t, "face": False, "yaw": 0, "pitch": 0, 
            "gaze_x": 0, "gaze_y": 0, "offscreen": False, "reading": 0, "emotion": "n/a"
        }

        # Обработка через MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks or len(res.multi_face_landmarks) == 0:
            return result

        # Взять лендмарки первого найденного лица (исправлено)
        lms = res.multi_face_landmarks[0].landmark
        result["face"] = True

        # Поза головы
        pts2d = np.array([[lms[LMS[k]].x*w, lms[LMS[k]].y*h] for k in LMS], dtype=np.float64)
        success, rvec, _ = cv2.solvePnP(MODEL_3D, pts2d, self.K, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            R, _ = cv2.Rodrigues(rvec)
            sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
            yaw = math.degrees(math.atan2(R[1,0], R[0,0]))
            pitch = math.degrees(math.atan2(-R[2,0], sy))
            result["yaw"] = self.prev_yaw = 0.7*self.prev_yaw + 0.3*yaw
            result["pitch"] = self.prev_pitch = 0.7*self.prev_pitch + 0.3*pitch

        # Вычисление взгляда
        def eye_gaze(eye_ids, iris_ids):
            eye = np.mean([[lms[i].x*w, lms[i].y*h] for i in eye_ids], axis=0)
            iris = np.mean([[lms[i].x*w, lms[i].y*h] for i in iris_ids], axis=0)
            return (iris - eye) / max(w, h)

        l_gaze = eye_gaze(LEFT_EYE, LEFT_IRIS)
        r_gaze = eye_gaze(RIGHT_EYE, RIGHT_IRIS)
        avg_gaze = (l_gaze + r_gaze) / 2
        result["gaze_x"] = float(avg_gaze[0])
        result["gaze_y"] = float(avg_gaze[1])

        # Проверка offscreen (смотрит в сторону слишком далеко)
        result["offscreen"] = abs(result["yaw"]) > 25 or abs(result["pitch"]) > 20

        # Выявление чтения глазами (движения взгляда)
        self.gaze_hist.append(result["gaze_x"])
        if len(self.gaze_hist) > 10:
            dx = np.diff(list(self.gaze_hist))
            result["reading"] = min(1.0, np.sum(dx > 0.001) / 5)

        return result

    def detect_emotion(self, frame):
        if not EMOTIONS_ON:
            return "n/a"
        try:
            res = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            # Иногда res - список, берем первый элемент
            if isinstance(res, list):
                res = res[0]
            emotion = res.get('dominant_emotion', 'neutral')
            self.emotions[emotion] = self.emotions.get(emotion, 0) + 1
            self.last_emotion = emotion
            return emotion
        except Exception:
            return self.last_emotion


def main():
    session_folder = create_folder()
    print(f"\n📁 Папка: {session_folder}")

    # Проверка камеры
    print("🎥 Проверка камеры...")
    cap = cv2.VideoCapture(0)

    for i in range(10):
        ret, frame = cap.read()
        if ret:
            print("✅ Камера работает!")
            break
        time.sleep(0.5)
    else:
        print("❌ ОШИБКА: Не удалось открыть камеру!")
        cap.release()
        return

    w = int(cap.get(3))
    h = int(cap.get(4))
    fps = cap.get(5) or 25.0
    print(f"📐 Разрешение: {w}x{h}, FPS: {fps}")
    print("\n📹 Нажмите 'q' для остановки\n")

    video_file = os.path.join(session_folder, "video.mp4")
    writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    analyzer = Analyzer()
    data = []
    t0 = time.time()
    frame_num = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            t = time.time() - t0
            frame_num += 1

            # Анализ
            res = analyzer.analyze(frame, t)
            res["frame"] = frame_num

            # Эмоции каждые 10 кадров
            if frame_num % 10 == 0:
                res["emotion"] = analyzer.detect_emotion(frame)
            else:
                res["emotion"] = analyzer.last_emotion

            data.append(res)

            # Визуализация
            if res["face"]:
                cv2.putText(frame, f"Yaw: {res['yaw']:.0f} Pitch: {res['pitch']:.0f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                cv2.putText(frame, f"Offscreen: {'YES' if res['offscreen'] else 'NO'}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0,0,255) if res['offscreen'] else (0,255,0), 2)
                cv2.putText(frame, f"Emotion: {res['emotion']}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                cv2.putText(frame, f"Reading: {res['reading']:.1f}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.putText(frame, f"REC {t:.1f}s | Frame {frame_num}", (w-200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.circle(frame, (w-220, 25), 5, (0,0,255), -1)

            writer.write(frame)
            cv2.imshow("Interview Analyzer (press 'q' to stop)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n⏹️ Остановка записи...")
                break

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    if not data:
        print("❌ Нет данных для анализа")
        return

    # Статистика
    total_time = time.time() - t0
    face_frames = [d for d in data if d["face"]]
    offscreen_pct = sum(1 for d in data if d["offscreen"]) / max(1, len(data))
    reading_avg = np.mean([d["reading"] for d in face_frames]) if face_frames else 0

    # Отчет
    report = {
        "duration_sec": round(total_time, 1),
        "frames": len(data),
        "face_detected_frames": len(face_frames),
        "offscreen_percent": round(offscreen_pct * 100, 1),
        "reading_score": round(reading_avg, 2),
        "emotions": dict(analyzer.emotions),
        "main_emotion": max(analyzer.emotions, key=analyzer.emotions.get) if analyzer.emotions else "n/a"
    }

    # Сохранение результатов
    with open(os.path.join(session_folder, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(os.path.join(session_folder, "data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print("\n" + "="*50)
    print(f"✅ Анализ завершен!")
    print(f"📁 Результаты: {session_folder}")
    print(f"⏱️ Длительность: {report['duration_sec']}с")
    print(f"📊 Кадров: {report['frames']} (лицо: {report['face_detected_frames']})")
    print(f"👀 Offscreen: {report['offscreen_percent']}%")
    print(f"📖 Reading: {report['reading_score']}")
    print(f"😊 Эмоция: {report['main_emotion']}")
    print("="*50)


if __name__ == "__main__":
    main()
