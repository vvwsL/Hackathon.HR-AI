import React, { useRef, useState } from "react";

export default function VideoInterview() {
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const [recording, setRecording] = useState(false);
  const [chunks, setChunks] = useState([]);
  const [transcript, setTranscript] = useState("");

  const start = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    videoRef.current.srcObject = stream;

    mediaRecorderRef.current = new MediaRecorder(stream);
    mediaRecorderRef.current.ondataavailable = e => setChunks(prev => [...prev, e.data]);
    mediaRecorderRef.current.start();
    setRecording(true);
  };

  const stop = () => {
    mediaRecorderRef.current.stop();
    setRecording(false);
  };

  const upload = async () => {
    const blob = new Blob(chunks, { type: "video/webm" });
    const formData = new FormData();
    formData.append("file", blob, "interview.webm");
    try {
      const res = await fetch("http://localhost:8000/upload_video", { method: "POST", body: formData });
      const data = await res.json();
      if(data.transcript){
        setTranscript(data.transcript);
      } else {
        setTranscript("Ошибка при распознавании");
      }
    } catch {
      setTranscript("Ошибка загрузки");
    }
  };

  return (
    <div>
      <h3>Видео-интервью</h3>
      <video ref={videoRef} autoPlay muted width={400} style={{ border: "1px solid #ccc" }} />
      {!recording ? <button onClick={start} style={{ marginTop: 10 }}>Начать запись</button> : <button onClick={stop} style={{ marginTop: 10 }}>Остановить запись</button>}
      <button onClick={upload} disabled={recording || chunks.length === 0} style={{ marginLeft: 10 }}>Отправить видео</button>
      {transcript && (
        <div style={{ marginTop: 20 }}>
          <h4>Распознанный текст:</h4>
          <p>{transcript}</p>
        </div>
      )}
    </div>
  );
}
