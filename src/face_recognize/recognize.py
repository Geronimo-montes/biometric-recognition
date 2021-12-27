#!/usr/bin/env python3
# coding=utf-8

import os
import cv2
import numpy as np

from face_recognize.face_recognize import detect_face

from utils.utils import H_FRAME
from utils.utils import PATH_DATA
from utils.utils import W_FRAME
from utils.utils import load_names_list

from utils.models import THRESHOLD
from utils.models import get_LBPHFaceRecognizer


def recognize_video():
    for name in os.listdir(PATH_DATA):
        file = os.path.join(PATH_DATA, name)

        fd = open(file, "rb")
        f = np.fromfile(fd, dtype=np.uint8, count=576 * 768)
        im = f.reshape((576, 768))
        fd.close()
        cv2.imshow("", im)

        cv2.waitKey(0) & 0xFF  # Press 'ESC' for exiting video
        cv2.destroyAllWindows()


def recognize_image(path_img: str):
    LBPHFaceRecognizer = get_LBPHFaceRecognizer()
    names = load_names_list()

    _img = cv2.imread(path_img)
    gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
    face = detect_face(gray)

    if not face:
        return "unknown", 0

    for (x, y, xx, yy) in face:
        img = np.array(gray[y:yy, x:xx], "uint8")
        id, confid = LBPHFaceRecognizer.predict(img)

        cv2.rectangle(_img, (x, y), (xx, yy), (0, 255, 0), 2)

        cv2.imshow("imagen", _img)
        cv2.waitKey(100)

        if confid < THRESHOLD:
            return names[id], round(100 - confid)
        else:
            return "unknown", 0


def recognize_webcam():
    LBPHFaceRecognizer = get_LBPHFaceRecognizer()
    names = load_names_list()

    cam = cv2.VideoCapture(0)
    cam.set(3, W_FRAME)  # set Width
    cam.set(4, H_FRAME)  # set Height

    while True:
        ret, frm = cam.read()
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        face = detect_face(gray)

        if not face:
            cv2.imshow("frame", frm)
            k = cv2.waitKey(10) & 0xFF  # Press 'ESC' for exiting video
            continue

        for (x, y, xx, yy) in face:
            img = np.array(gray[y:yy, x:xx], "uint8")
            id, confid = LBPHFaceRecognizer.predict(img)

            if confid < THRESHOLD:
                id = names[id]
                color = (0, 255, 0)  # verde
            else:
                id = "unknown"
                color = (0, 0, 255)  # rojo

            confid = "{:3.1f}%".format(round(100 - confid))
            cv2.rectangle(frm, (x, y), (xx, yy), color, 2)
            cv2.putText(frm, str(id), (x + 5, y - 5), 2, 1, color, 4)
            cv2.putText(frm, str(confid), (x + 5, yy - 5), 2, 1, color, 2)

        cv2.imshow("frame", frm)
        k = cv2.waitKey(10) & 0xFF  # Press 'ESC' for exiting video
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
