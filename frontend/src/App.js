import React from 'react';
import Register from './components/Register';
import ResumeUpload from './components/ResumeUpload';
import VideoInterview from './components/VideoInterview';
import DynamicVoiceInterview from './components/DynamicVoiceInterview';  // <-- импорт

export default function App() {
  return (
    <div style={{ padding: 20 }}>
      <h1>AI HR Platform</h1>
      <Register />
      <hr />
      <ResumeUpload />
      <hr />
      <VideoInterview />
      <hr />
      <DynamicVoiceInterview />   {/* <-- добавьте сюда для отображения */}
    </div>
  );
}
