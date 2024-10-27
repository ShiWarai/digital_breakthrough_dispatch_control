import time
import os
import threading
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_from_directory
from ultralytics import YOLO
import warnings
import supervision as sv
from werkzeug.utils import secure_filename
from typing import Callable
import cv2
import shutil
from inference import get_model
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder='assets')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable file caching if needed
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['STATIC_FOLDER'] = 'assets'
processing_complete = dict()


model_trains = YOLO("models/best.pt")
model_danger = get_model(model_id="final-zjnyf/5", api_key=os.getenv('ROBOFLOW_KEY'))


class ObjectTracker:
    def __init__(self, max_distance=30):
        self.max_distance = max_distance
        self.tracked_objects = {}
        self.next_id = 1
        self.object_classes = {}

    def update(self, detections):
        current_detections = detections.xyxy
        for i, detection in enumerate(current_detections):
            x1, y1, x2, y2 = detection
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            min_distance = float('inf')
            min_id = None
            for obj_id, (prev_center_x, prev_center_y) in self.tracked_objects.items():
                distance = np.sqrt((center_x - prev_center_x) ** 2 + (center_y - prev_center_y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    min_id = obj_id

            if min_distance < self.max_distance:
                self.tracked_objects[min_id] = (center_x, center_y)
                obj_class = detections.data['class_name'][i]
                self.object_classes[min_id] = obj_class
            else:
                self.tracked_objects[self.next_id] = (center_x, center_y)
                obj_class = detections.data['class_name'][i]
                self.object_classes[self.next_id] = obj_class
                self.next_id += 1

    def get_object_counts(self):
        class_counts = {}
        for obj_class in self.object_classes.values():
            if obj_class in class_counts:
                class_counts[obj_class] += 1
            else:
                class_counts[obj_class] = 1
        return class_counts

final_count = {}

def clear_upload_folder(upload_folder):
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Не удалось удалить {file_path}. Причина: {e}')

def generate_graph(results: dict[str, int], filename: str):
    plt.title('Вывод статистики')

    labels = list(results.keys())
    sizes = list(results.values())

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')

    plt.savefig(os.path.join(app.config['OUTPUT_FOLDER'], filename+'_graph.png'))


def time_danger_graph(data: list[dict[str, int]], filename: str):
    if len(data) == 0:
        pass

    # Количество кадров
    total_frames = len(data)
    # Расчет кадров в секунду (FPS)
    fps = 30
    # Получение данных только для 'person'
    person_counts = [frame.get('person', 0) for frame in data]

    # Создание списка секунд, соответствующих каждому кадру
    seconds = [i / fps for i in range(total_frames)]

    # Создание графика
    plt.figure(figsize=(10, 5))
    plt.plot(seconds, person_counts, marker='o', linestyle='-', color='b')

    # Добавление названий осей и заголовка
    plt.xlabel('Протяженность видео')
    plt.ylabel('Количество опасностей на путях')
    plt.title('Временной ряд опасностей на Ж/Д путях')
    plt.legend()
    plt.grid(True)

    # Настройка оси Y для отображения только целых чисел
    plt.yticks(
        range(0, max(person_counts) + 2))  # Устанавливаем метки с шагом 1, включая максимальное значение + 1 для маржи

    # Показать график
    plt.savefig(os.path.join(app.config['OUTPUT_FOLDER'], filename+'_danger_graph.png'))

def process_frame(frame: np.ndarray, index: int, tracker_trains: ObjectTracker, tracker_danger: ObjectTracker, by_frame_count: list) -> np.ndarray:
    results_danger = model_danger.infer(frame)[0]
    results_trains = model_trains(frame, imgsz=1280)[0]

    detections_danger = sv.Detections.from_inference(results_danger)
    detections_trains = sv.Detections.from_ultralytics(results_trains)

    tracker_danger.update(detections_danger)
    tracker_trains.update(detections_trains)

    if (tracker_danger.get_object_counts() != 0):
        by_frame_count.append(tracker_danger.get_object_counts())
    print("New person detection:", tracker_danger.get_object_counts())
    print("New trains detection:", tracker_trains.get_object_counts())

    box_annotator = sv.BoundingBoxAnnotator()

    label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)

    annotated_image = frame.copy()
    annotated_image = box_annotator.annotate(annotated_image, detections=detections_danger)
    annotated_image = label_annotator.annotate(annotated_image, detections=detections_danger)
    annotated_image = box_annotator.annotate(annotated_image, detections=detections_trains)
    annotated_image = label_annotator.annotate(annotated_image, detections=detections_trains)
    return annotated_image


def process_video(video_path: str, output_path: str, callback: Callable[[np.ndarray, int, ObjectTracker, ObjectTracker, list], np.ndarray]) -> None:
    global final_count, processing_complete

    filename = os.path.basename(output_path)
    tracker_trains = ObjectTracker()
    tracker_danger = ObjectTracker()
    by_frame_count = []

    # Video processing logic
    video_info = sv.VideoInfo.from_video_path(video_path)
    with sv.VideoSink(target_path=output_path, video_info=video_info) as sink:
        for index, frame in enumerate(sv.get_video_frames_generator(source_path=video_path, stride=50)):
            result_frame = callback(frame, index, tracker_trains, tracker_danger, by_frame_count)
            sink.write_frame(frame=result_frame)

    final_count = [tracker_trains.get_object_counts(), tracker_danger.get_object_counts()]
    generate_graph(tracker_trains.get_object_counts(), filename)
    time_danger_graph(by_frame_count, filename)

    print("Graph: ", os.path.join(app.config['OUTPUT_FOLDER'], filename+'_graph.png'))

    processing_complete[output_path] = True


@app.route('/upload', methods=['POST'])
def upload_file():
    global processing_complete

    file = request.files['video']

    if file:
        filename = secure_filename(file.filename)

        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        file.save(video_path)

        processing_complete[output_path] = False

        threading.Thread(target=process_video, args=(video_path, output_path, process_frame), daemon=True).start()

        return jsonify({'video_url': url_for('output_video', video=filename), 'final_count_url': url_for('final_count'),
                        'status_url': url_for('processing_status', video=filename), 'graph_url': url_for('graph', video=filename),
                        'danger_graph_url': url_for('danger_graph', video=filename)})
    return redirect(url_for('index'))


@app.route('/processing_status/<video>')
def processing_status(video):
    return jsonify({'processing_complete': processing_complete[os.path.join(app.config['OUTPUT_FOLDER'], video)]})


def generate_frames(video_path, delay=0.1):
    while processing_complete[video_path]:
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                time.sleep(delay)

        cap.release()


@app.route('/output_video/<video>')
def output_video(video: str):
    return Response(generate_frames(os.path.join(app.config['OUTPUT_FOLDER'], video)), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/final_count')
def final_count():
    return jsonify(final_count)

@app.route('/graph/<video>')
def graph(video: str):
    filename = f"{video}_graph.png"
    return send_from_directory(directory=app.config['OUTPUT_FOLDER'], path=filename)

@app.route('/danger_graph/<video>')
def danger_graph(video: str):
    filename = f"{video}_danger_graph.png"
    return send_from_directory(directory=app.config['OUTPUT_FOLDER'], path=filename)


@app.route('/')
def index():
    return render_template('index.html')


clear_upload_folder(app.config['UPLOAD_FOLDER'])
clear_upload_folder(app.config['OUTPUT_FOLDER'])

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host="0.0.0.0", port=8080)
