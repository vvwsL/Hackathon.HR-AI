import React, { useState } from "react";
import axios from "axios";

export default function Register() {
  const [form, setForm] = useState({ username: "", email: "", password: "", role: "employee" });
  const [msg, setMsg] = useState("");

  const onChange = e => setForm({ ...form, [e.target.name]: e.target.value });

  const onSubmit = async e => {
    e.preventDefault();
    try {
      await axios.post("http://localhost:8000/register", form);
      setMsg("Регистрация успешна");
    } catch {
      setMsg("Ошибка регистрации");
    }
  };

  return (
    <form onSubmit={onSubmit}>
      <input name="username" placeholder="Логин" onChange={onChange} required />
      <input name="email" type="email" placeholder="Email" onChange={onChange} required />
      <input name="password" type="password" placeholder="Пароль" onChange={onChange} required />
      <select name="role" onChange={onChange} value={form.role}>
        <option value="employee">Соискатель</option>
        <option value="hr">HR</option>
      </select>
      <button type="submit" style={{ marginTop: 10 }}>Зарегистрироваться</button>
      <p>{msg}</p>
    </form>
  );
}
