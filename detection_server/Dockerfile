# Используем официальный образ Python
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

WORKDIR /app

# Копируем файл requirements.txt в контейнер
COPY requirements.txt .

RUN apt-get install libgl1-mesa-glx

# Устанавливаем зависимости из requirements.txt
RUN pip install -r requirements.txt

# Копируем содержимое папки detection_server в контейнер
COPY . .

# Определяем команду для запуска скрипта
CMD waitress-serve --host 0.0.0.0 --port 8080 main:app