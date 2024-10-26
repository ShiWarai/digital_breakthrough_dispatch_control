# Используем официальный образ Python
FROM python:3.12

WORKDIR /app

# Копируем файл requirements.txt в контейнер
COPY requirements.txt .

# Устанавливаем зависимости из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем содержимое папки detection_server в контейнер
COPY . .

# Определяем команду для запуска скрипта
CMD ["python", "main.py"]