#!/usr/bin/env python3
# coding=utf-8

import os
import shutil

import cv2
import numpy as np
from cv2 import VideoCapture
from model.decorators import validator_method_add

from model.train import train

from model.recognize import detect_face

from utils.settings import PATH_DATABASE
from utils.settings import H_FRAME
from utils.settings import W_FRAME

from utils.settings import save_names_list
from utils.settings import set_new_id


@validator_method_add(["name", "path_dir"])
def add_galery(name: str, path_dir: str):
    count = 0
    id = set_new_id(name)
    name = name.replace("_", "-")
    _faces, _ids = [], []

    for img in os.listdir(path_dir):
        path_img = os.path.join(path_dir, img)

        _img = cv2.imread(path_img)
        # _img = cv2.resize(_img, (0, 0), fx=1.2, fy=1.2, interpolation=cv2.INTER_NEAREST)

        gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)

        print("Detectando Rostro...", flush=True, end="")
        face = detect_face(gray)

        if not face:
            # print("Rostro no identificado...", flush=True, end="")
            raise Exception("NOT FACE DETECT")

        else:
            for (x, y, xx, yy) in face:
                print("Save img...", flush=True, end="")
                count += 1
                img = gray[y:yy, x:xx]

                _faces.append(img)
                _ids.append(id)

                cv2.imwrite(
                    os.path.join(PATH_DATABASE, f"{id}_{name}_{count}.jpg"), img
                )
                cv2.rectangle(_img, (x, y), (xx, yy), (0, 255, 0), 2)

        cv2.imshow("imagen", _img)
        k = cv2.waitKey(100) & 0xFF  # Press 'ESC' for exiting video

    cv2.destroyAllWindows()
    train((_faces, np.array(_ids)))

    return name, count, id


@validator_method_add(["name", "path_dir"])
def add_galery_to_db(name: str, path_dir: str):
    count = 0
    id = set_new_id(name)
    name = name.replace("_", "-")
    _faces, _ids = [], []

    for img in os.listdir(path_dir):
        src = os.path.join(path_dir, img)
        dst = os.path.join(PATH_DATABASE, f"{id}_{name}_{count}.jpg")
        print("Save img...", flush=True, end="")
        count += 1
        shutil.move(src, dst)

    train()

    return name, count, id


@validator_method_add(["name"])
def add_webcam(name: str):
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
