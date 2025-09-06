import React, { useState } from "react";
import axios from "axios";

export default function ResumeUpload() {
  const [file, setFile] = useState(null);
  const [score, setScore] = useState(null);

  const onFileChange = e => setFile(e.target.files[0]);

  const onUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await axios.post("http://localhost:8000/upload_resume", formData);
      setScore(res.data.score || "Ошибка анализа");
    } catch {
      setScore("Ошибка загрузки");
    }
  };

  return (
    <div>
      <h3>Загрузка резюме</h3>
      <input type="file" accept=".pdf" onChange={onFileChange} />
      <button onClick={onUpload} style={{ marginLeft: 10 }}>Загрузить</button>
      {score !== null && <p>Результат анализа: {score}</p>}
    </div>
  );
}
