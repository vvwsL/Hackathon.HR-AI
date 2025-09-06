import React, { useState, useEffect } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';

const questions = [
  "Расскажите, пожалуйста, о вашем опыте работы с Python.",
  "Какие проекты с Django вы реализовывали?",
  "Как вы оцениваете свои знания SQL?",
];

export default function DynamicVoiceInterview() {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState([]);

  const {
    transcript,
    resetTranscript,
    listening,
    browserSupportsSpeechRecognition
  } = useSpeechRecognition();

  useEffect(() => {
    if (transcript && transcript.length > 0) {
      setAnswers(prev => [
        ...prev,
        { question: questions[currentQuestion], answer: transcript }
      ]);
      resetTranscript();
      setTimeout(() => {
        setCurrentQuestion(i => i + 1);
        SpeechRecognition.stopListening();
      }, 500);
    }
  }, [transcript]);

  const startListening = () => {
    SpeechRecognition.startListening({
      continuous: false,
      language: 'ru-RU'
    });
  };

  if (!browserSupportsSpeechRecognition) {
    return <p>Ваш браузер не поддерживает распознавание речи.</p>;
  }

  if (currentQuestion >= questions.length) {
    return (
      <div>
        <h3>Интервью завершено!</h3>
        <ul>
          {answers.map((qa, idx) => (
            <li key={idx}><b>{qa.question}</b><br />{qa.answer}</li>
          ))}
        </ul>
      </div>
    );
  }

  return (
    <div>
      <h4>{questions[currentQuestion]}</h4>
      <button onClick={startListening} disabled={listening}>
        {listening ? "Слушаю..." : "Ответить голосом"}
      </button>
      <p><i>Ваш ответ: {transcript}</i></p>
    </div>
  );
}
