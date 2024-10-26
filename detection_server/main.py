import os
import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import PIL.Image, PIL.ImageTk
from ultralytics import YOLO
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

# Загрузите вашу модель
model = YOLO("best.pt")

window = tk.Tk()
window.title("Видеообработчик")
window.geometry("1000x600")  # Начальный размер окна

canvas = tk.Canvas(window, width=640, height=480)
canvas.pack(side=tk.RIGHT)

progress = ttk.Progressbar(window, orient="horizontal", length=300, mode='determinate')
progress.pack(side=tk.TOP, padx=20, pady=20)


def load_video():
    file_path = filedialog.askopenfilename()
    if file_path:
        update_video_date(file_path)
        threading.Thread(target=lambda: process_video(file_path, "output.mp4", process_frame), daemon=True).start()


previous_frame = {}
final_count = {}


def process_frame(frame: np.ndarray, test: int) -> np.ndarray:
    global previous_frame
    results = model(frame, imgsz=1280)[0]

    detections = sv.Detections.from_ultralytics(results)
    print(detections.data['class_name'])
    lst = detections.data['class_name']
    a = list(map(lambda x: x.replace("np.str_('", '').replace("')", ''), lst))
    dictionary = {}
    for item in a:
        dictionary[item] = dictionary.get(item, 0) + 1
    if dictionary == previous_frame:
        print(True)
    else:
        set1 = set(dictionary.items())
        set2 = set(previous_frame.items())
        diff_res = list(set1 ^ set2)
        logger.info(diff_res)
        print(diff_res)
        if (len(diff_res) > 1): # То есть прям жесткая ошибка пошла, два класса полетело
            for item in diff_res:
                if item[0] in final_count:
                    final_count[item[0]] = final_count[item[0]] + item[-1]
                else:
                    final_count[item[0]] = item[-1]
            print("CRITICAL")
        elif (len(diff_res) == 1) and (diff_res[0][-1] > 2): # или в одном классе появилось много вагонов:
            final_count[[0][0]] = diff_res[0][-1]
            print("CRITICAL")

    logger.info(dictionary)
    print(dictionary)
    print(test)
    print("out of " + str(final_count))
    previous_frame = dictionary
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
    VideoInfo = sv.VideoInfo.from_video_path(source_path)
    source_video_info = VideoInfo.from_video_path(video_path=source_path)
    frame_count = int(cv2.VideoCapture(source_path).get(cv2.CAP_PROP_FRAME_COUNT))
    progress['maximum'] = frame_count/50 # stride
    with sv.VideoSink(target_path=target_path, video_info=source_video_info) as sink:
        for index, frame in enumerate(
            sv.get_video_frames_generator(source_path=source_path, stride=50) #50 frames skip
        ):
            result_frame = callback(frame, index)
            if index % sink.video_info.fps == 0:
                process_frame(frame, index + 1) # Номер кадра
            sink.write_frame(frame=result_frame)
            window.after(50, update_progress, index + 1)  # Обновляем прогрессбар в основном потоке

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
