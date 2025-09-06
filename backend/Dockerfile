FROM python:3.9-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Сначала установите базовые зависимости
COPY requirements_basic.txt .
RUN pip install --no-cache-dir -r requirements_basic.txt

# Затем установите ML зависимости
COPY requirements_ml.txt .
RUN pip install --no-cache-dir -r requirements_ml.txt

# Дополнительные зависимости для обработки документов
RUN pip install --no-cache-dir pypandoc==1.11

# Копирование исходного кода
COPY . .

# Создание необходимых директорий
RUN mkdir -p sessions uploads jobs reports

EXPOSE 5000

CMD ["python", "app.py"]
