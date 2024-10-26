import time
import os
import threading
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from ultralytics import YOLO
import supervision as sv
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from typing import Callable
import cv2
import shutil

# # Configure logging
# logging.basicConfig(filename="EXAMPLE.log",
#                     filemode='a',
#                     format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
#                     datefmt='%H:%M:%S',
#                     level=logging.DEBUG)
# logger = logging.getLogger('urbanGUI')
# logging.info("Running Urban Planning")

app = Flask(__name__, static_folder='static')
app.wsgi_app = ProxyFix(app.wsgi_app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable file caching if needed
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['STATIC_FOLDER'] = 'static/videos'
processing_complete = False  # Global flag for processing status


model = YOLO("models/best.pt")


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


tracker = ObjectTracker()
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

def process_frame(frame: np.ndarray, index: int) -> np.ndarray:
    global tracker
    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_ultralytics(results)
    tracker.update(detections)

    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)
    annotated_image = frame.copy()
    annotated_image = box_annotator.annotate(annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(annotated_image, detections=detections)
    return annotated_image


def process_video(video_path: str, output_path: str, callback: Callable[[np.ndarray, int], np.ndarray]) -> None:
    global final_count, processing_complete
    processing_complete = False

    # Video processing logic
    video_info = sv.VideoInfo.from_video_path(video_path)
    with sv.VideoSink(target_path=output_path, video_info=video_info) as sink:
        for index, frame in enumerate(sv.get_video_frames_generator(source_path=video_path, stride=50)):
            result_frame = callback(frame, index)
            sink.write_frame(frame=result_frame)

    final_count = tracker.get_object_counts()
    processing_complete = True

    if os.path.exists(video_path):
        os.remove(video_path)


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['video']
    if file:
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.mp4')
        file.save(video_path)

        threading.Thread(target=process_video, args=(video_path, output_path, process_frame), daemon=True).start()

        return jsonify({'video_url': url_for('output_video'), 'final_count_url': url_for('final_count'),
                        'status_url': url_for('processing_status')})
    return redirect(url_for('index'))


@app.route('/processing_status')
def processing_status():
    return jsonify({'processing_complete': processing_complete})


def generate_frames(video_path, delay=0.1):
    while True:
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


@app.route('/output_video')
def output_video():
    return Response(generate_frames('output/output.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/final_count')
def final_count():


    return jsonify(final_count)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    clear_upload_folder(app.config['UPLOAD_FOLDER'])
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

    app.run(debug=True, threaded=True, port=8080)
