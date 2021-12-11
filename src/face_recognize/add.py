#!/usr/bin/env python3
# coding=utf-8

import os
import cv2
import numpy as np
from PIL import Image
from cv2 import VideoCapture

from face_recognize.train import train

from face_recognize.face_recognize import detect_face

from utils.utils import PATH_DATABASE
from utils.utils import H_FRAME
from utils.utils import W_FRAME

from utils.utils import save_names_list
from utils.utils import set_new_id


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
        path_img = os.path.join(path_dir, img)

        _img = cv2.imread(path_img)
        # _img = cv2.resize(_img, (0, 0), fx=1.2, fy=1.2, interpolation=cv2.INTER_NEAREST)

        gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)

        print("Detectando Rostro...")
        face = detect_face(gray)

        if not face:
            print("Rostro no identificado...")
        else:
            for (x, y, xx, yy) in face:
                print("Save img...")
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
