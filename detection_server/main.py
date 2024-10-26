import os
import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import PIL.Image, PIL.ImageTk
from inference import get_model

import supervision as sv
from typing import Callable
import numpy as np
import warnings
import threading
import os.path, time, locale
import sys

warnings.filterwarnings("ignore")

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger("ultralytics").setLevel(logging.DEBUG)
logging.basicConfig(filename="EXAMPLE.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.info("Running Urban Planning")

logger = logging.getLogger('urbanGUI')


locale.setlocale(locale.LC_ALL, 'Russian_Russia.1251')

model = get_model(model_id="final-zjnyf/5", api_key="oMwwId6tzG8Aga5aGVo2")


window = tk.Tk()
window.title("Видеообработчик")
window.geometry("1000x600")  # Начальный размер окна

canvas = tk.Canvas(window, width=640, height=480)
canvas.pack(side=tk.RIGHT)

progress = ttk.Progressbar(window, orient="horizontal", length=300, mode='determinate')
progress.pack(side=tk.TOP, padx=20, pady=20)

class ObjectTracker:
    def __init__(self, max_distance=50):
        self.max_distance = max_distance
        self.tracked_objects = {}
        self.next_id = 1
        self.object_classes = {}

    def update(self, detections):
        current_detections = detections.xyxy
        danger_flag = False
        if len(list(current_detections)) > 2: # Если больше 3 людей в моменте для наглядности
            print(f"Danger, {len(list(current_detections))} people")
            danger_flag = True

        for i, detection in enumerate(current_detections):
            x1, y1, x2, y2 = detection
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Найти ближайший объект
            min_distance = float('inf')
            min_id = None
            for obj_id, (prev_center_x, prev_center_y) in self.tracked_objects.items():
                distance = np.sqrt((center_x - prev_center_x) ** 2 + (center_y - prev_center_y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    min_id = obj_id

            # Если ближайший объект находится в пределах допустимого расстояния, обновляем его
            if min_distance < self.max_distance:
                self.tracked_objects[min_id] = (center_x, center_y)
                obj_class = detections.data['class_name'][i]
                self.object_classes[min_id] = obj_class
            else:
                # Иначе создаем новый объект
                self.tracked_objects[self.next_id] = (center_x, center_y)
                obj_class = detections.data['class_name'][i]
                self.object_classes[self.next_id] = obj_class
                self.next_id += 1
        return danger_flag

    def get_object_counts(self):
        class_counts = {}
        for obj_class in self.object_classes.values():
            if obj_class in class_counts:
                class_counts[obj_class] += 1
            else:
                class_counts[obj_class] = 1
        return class_counts

previous_frame = {}
final_count = {}
tracker = ObjectTracker()

def load_video():
    file_path = filedialog.askopenfilename()
    if file_path:
        update_video_date(file_path)
        threading.Thread(target=lambda: process_video(file_path, "digital_breakthrough_dispatch_control/output.mp4", process_frame), daemon=True).start()


def process_frame(frame: np.ndarray, test: int) -> np.ndarray:
    global previous_frame
    global tracker

    results = model.infer(frame)[0]

    detections = sv.Detections.from_inference(results)

    danger = tracker.update(detections)
    if (danger):
        print(f"Danger detected at {test} frame, thats {test/70} second") # fps

    print("New detection:", tracker.get_object_counts())

    box_annotator = sv.BoundingBoxAnnotator()

    label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)

    annotated_image = frame.copy()
    annotated_image = box_annotator.annotate(annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(annotated_image, detections=detections)
    return annotated_image


def process_video(
    source_path: str,
    target_path: str,
    callback: Callable[[np.ndarray, int], np.ndarray],
) -> None:
    global previous_frame
    global final_count

    previous_frame = {}
    final_count = {}

    VideoInfo = sv.VideoInfo.from_video_path(source_path)
    source_video_info = VideoInfo.from_video_path(video_path=source_path)
    frame_count = int(cv2.VideoCapture(source_path).get(cv2.CAP_PROP_FRAME_COUNT))
    progress['maximum'] = frame_count/20 # stride
    with sv.VideoSink(target_path=target_path, video_info=source_video_info) as sink:
        for index, frame in enumerate(
            sv.get_video_frames_generator(source_path=source_path, stride=20) #20 frames skip
        ):
            result_frame = callback(frame, index)
            if index % sink.video_info.fps == 0:
                process_frame(frame, index + 1) # Номер кадра
            sink.write_frame(frame=result_frame)
            window.after(50, update_progress, index + 1)  # Обновляем прогрессбар в основном потоке

    final_count = tracker.get_object_counts()
    print('-'*40)
    print("Кол-во объектов: ", final_count)
    print('-' * 40)

    display_video('output.mp4')


photo = None  # Глобальная переменная для хранения текущего фото


def display_video(path):
    global photo  # Объявляем переменную глобальной
    update_video_date(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return

    def stream():
        global photo
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (canvas.winfo_width(), canvas.winfo_height()), interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            window.after(100, stream)
        else:
            cap.release()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            display_video(path)
    stream()


# Label для даты создания видео
date_label = tk.Label(window, text=" ")
date_label.pack(side=tk.BOTTOM, fill=tk.X)


def update_video_date(file_path):
    ti_c = os.path.getctime(file_path)
    # Converting the time in seconds to a timestamp
    format = "%a, %d %b %Y %H:%M:%S"  # строка для нужного форматирования
    c_ti = time.strftime(format, time.localtime(ti_c))
    # Видео идет 2 минуты
    date_label.config(text=f"Дата создания - {c_ti} | Продолжительность - 2 минуты")


def update_progress(value):
    progress['value'] = value


load_btn = tk.Button(window, text="Загрузить видео", command=load_video)
load_btn.pack(side=tk.LEFT, padx=20, pady=20)

window.mainloop()
