import cv2
import mediapipe as mp
import numpy as np
import time
import json
import threading
import queue
import os
from datetime import datetime
import sounddevice as sd
import soundfile as sf
import wave
from deepface import DeepFace
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# ==== Настройки ====
SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 20

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

frame_queue = queue.Queue(maxsize=5)
emotion_queue = queue.Queue()
stop_event = threading.Event()
audio_queue = queue.Queue()


def create_folder():
    folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def audio_callback(indata, frames, time_info, status):
    audio_queue.put(bytes(indata))


def record_audio(duration_sec, folder):
    audio_data = b''
    audio_levels = []  # будем логировать уровень громкости по времени
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype='int16',
                          channels=CHANNELS, callback=audio_callback):
        start_time = time.time()
        while time.time() - start_time < duration_sec:
            try:
                audio_chunk = audio_queue.get(timeout=1)
                audio_data += audio_chunk
                # уровень громкости (RMS)
                lvl = np.sqrt(np.mean(np.frombuffer(audio_chunk, np.int16)**2))
                audio_levels.append({"time": time.time()-start_time, "level": float(lvl)})
            except queue.Empty:
                pass
    wav_path = os.path.join(folder, "audio.wav")
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data)
    return wav_path, audio_levels


def detect_emotion_thread():
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            frame, timestamp = frame_queue.get(timeout=1)
        except queue.Empty:
            continue
        try:
            result = DeepFace.analyze(frame, actions=['emotion'],
                                      enforce_detection=False, silent=True)
            if isinstance(result, list):
                result = result[0]
            emotion = result.get("dominant_emotion", "neutral")
        except Exception:
            emotion = "neutral"
        emotion_queue.put((timestamp, emotion))


def compute_gaze(lms, w, h):
    def mean_point(ids):
        return np.mean([[lms[i].x * w, lms[i].y * h] for i in ids], axis=0)
    left_eye_center = mean_point(LEFT_EYE)
    left_iris_center = mean_point(LEFT_IRIS)
    right_eye_center = mean_point(RIGHT_EYE)
    right_iris_center = mean_point(RIGHT_IRIS)
    gaze_left = (left_iris_center - left_eye_center) / max(w, h)
    gaze_right = (right_iris_center - right_eye_center) / max(w, h)
    avg_gaze = (gaze_left + gaze_right) / 2
    return float(avg_gaze[0]), float(avg_gaze[1])


def load_wav2vec_model():
    processor = Wav2Vec2Processor.from_pretrained(
        "jonatasgrosman/wav2vec2-large-xlsr-53-russian")
    model = Wav2Vec2ForCTC.from_pretrained(
        "jonatasgrosman/wav2vec2-large-xlsr-53-russian")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device


def transcribe_audio(processor, model, device, wav_path):
    speech, sample_rate = sf.read(wav_path)
    inputs = processor(speech, sampling_rate=sample_rate,
                       return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription.lower()


def run_interview(question_text):
    folder = create_folder()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 25)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Видеозапись
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_path = os.path.join(folder, "video.avi")
    out = cv2.VideoWriter(video_path, fourcc, 25.0, (640, 480))

    for _ in range(10):
        cap.read()

    thread_emo = threading.Thread(target=detect_emotion_thread)
    thread_emo.start()

    # Запуск аудиозаписи
    audio_result = {}
    def audio_func():
        path, levels = record_audio(RECORD_SECONDS, folder)
        audio_result["wav"] = path
        audio_result["levels"] = levels
    audio_thread = threading.Thread(target=audio_func)
    audio_thread.start()

    print("Вопрос:", question_text)
    start_time = time.time()
    emotions, gazes = [], []
    last_emotion = "neutral"
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        t = time.time() - start_time
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        gaze_x, gaze_y = 0, 0
        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            if frame_counter % 10 == 0:
                if frame_queue.full():
                    frame_queue.get_nowait()
                frame_queue.put((frame.copy(), t))

            gaze_x, gaze_y = compute_gaze(lms, w, h)
            gazes.append({"time": round(t, 2), "gaze_x": gaze_x, "gaze_y": gaze_y})

            try:
                while True:
                    last_emotion = emotion_queue.get_nowait()[1]
            except queue.Empty:
                pass

            emotions.append({"time": round(t, 2), "emotion": last_emotion})
        out.write(frame)
        cv2.imshow("Видео", frame)

        if cv2.waitKey(1) & 0xFF == 27 or t > RECORD_SECONDS:
            break
        frame_counter += 1

    stop_event.set()
    thread_emo.join()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    audio_thread.join()

    processor, model, device = load_wav2vec_model()
    transcription = transcribe_audio(processor, model, device, audio_result["wav"])

    # Логика выявления заминок: длинные паузы ( >1.5 сек тишины )
    pauses = [lvl["time"] for lvl in audio_result["levels"] if lvl["level"] < 50]
    long_pauses = []
    if pauses:
        # ищем "пробелы" времени без звука
        last = pauses[0]
        cnt = 1
        for t in pauses[1:]:
            if t - last < 0.5:  # слитные тишины
                cnt += 1
            else:
                if cnt > 10:  # ~1.5 сек
                    long_pauses.append(last)
                cnt = 1
            last = t

    suspicious = len(long_pauses) > 0 or transcription.count("ээ") > 2

    result = {
        "question": question_text,
        "analysis": {
            "speech": transcription,
            "emotions": emotions,
            "gaze": gazes,
            "pauses": long_pauses,
            "suspicious": suspicious,
            "video": video_path,
            "attention": "ok" if len(gazes) > 5 else "no_face"
        }
    }

    with open(os.path.join(folder, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Результат в JSON:", folder)
    return result
