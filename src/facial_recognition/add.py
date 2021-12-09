#!/usr/bin/env python3
# coding=utf-8

import os
import cv2
import numpy as np
from PIL import Image
from cv2 import VideoCapture

from facial_recognition.train import train
from facial_recognition.detect_face import detect_face

from utils.utils import PATH_DATABASE, save_names_list, set_new_id
from utils.utils import H_FRAME, W_FRAME


def valid():
    imgs = os.listdir(PATH_DATABASE)
    if len(imgs) < 1:
        save_names_list([])


def add_from_galery(name: str, path_dir: str):
    valid()
    count = 0
    id = set_new_id(name)
    _faces, _ids = [], []

    for img in os.listdir(path_dir):
        imagen = Image.open(os.path.join(path_dir, img)).convert("L").reduce(factor=8)
        img_np = np.array(imagen, "uint8")
        face = detect_face(img_np)

        if not face:
            continue

        for (x, y, xx, yy) in face:
            count += 1
            img = img_np[y:yy, x:xx]
            _faces.append(img)
            _ids.append(id)
            cv2.imwrite(os.path.join(PATH_DATABASE, f"{id}_{name}_{count}.jpg"), img)

    train((_faces, np.array(_ids)))


def add_from_webcam(name: str):
    valid()
    count = 0
    id = set_new_id(name)
    _faces, _ids = [], []

    cam = VideoCapture(0)
    cam.set(3, W_FRAME)  # set Width
    cam.set(4, H_FRAME)  # set Height

    while True:
        ret, frame = cam.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = detect_face(frame_gray)

        if not face:
            cv2.imshow("frame", frame)
            k = cv2.waitKey(100) & 0xFF  # Press 'ESC' for exiting video
            continue

        for (x, y, xx, yy) in face:
            count += 1
            img = frame_gray[y:yy, x:xx]
            _faces.append(img)
            _ids.append(id)
            cv2.imwrite(os.path.join(PATH_DATABASE, f"{id}_{name}_{count}.jpg"), img)
            cv2.rectangle(frame, (x, y), (xx, yy), (0, 255, 0), 2)

        cv2.imshow("frame", frame)
        k = cv2.waitKey(100) & 0xFF  # Press 'ESC' for exiting video
        if k == 27 or count >= 10:
            break

    cam.release()
    cv2.destroyAllWindows()

    train((_faces, np.array(_ids)))
