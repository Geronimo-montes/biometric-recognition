#!/usr/bin/env python3
# coding=utf-8

import os
import cv2
import numpy as np

from PIL import Image
from facial_recognition.detect_face import detect_face

from utils.utils import H_FRAME, PATH_INPUT, W_FRAME
from utils.utils import get_names_ids
from utils.utils import get_LBPHFaceRecognizer


def recognize_video():
    for name in os.listdir(PATH_INPUT):
        file = os.path.join(PATH_INPUT, name)

        fd = open(file, "rb")
        f = np.fromfile(fd, dtype=np.uint8, count=576 * 768)
        im = f.reshape((576, 768))
        fd.close()
        cv2.imshow("", im)

        cv2.waitKey(0) & 0xFF  # Press 'ESC' for exiting video
        cv2.destroyAllWindows()
    # frm = cv2.imread(file)
    # cv2.imshow("picture", frm)


def recognize_image(path_img: str):
    LBPHFaceRecognizer = get_LBPHFaceRecognizer()
    ids, names = get_names_ids()

    imagen = Image.open(path_img).convert("L").reduce(factor=8)
    gray = np.array(imagen, "uint8")
    face = detect_face(gray)

    if not face:
        print("None")
    else:
        for (x, y, xx, yy) in face:
            _gray = gray[y:yy, x:xx]
            id, confid = LBPHFaceRecognizer.predict(_gray)

            if confid < 30:
                id = names[id]
                color = (0, 255, 0)  # verde
            else:
                id = "unknown"
                color = (0, 0, 255)  # rojo

            confid = "{:3.1f}%".format(round(100 - confid))
            cv2.rectangle(gray, (x, y), (xx, yy), color, 2)
            cv2.putText(gray, str(id), (x + 5, y - 5), 2, 1, color, 4)
            cv2.putText(gray, str(confid), (x + 5, yy - 5), 2, 1, color, 2)

        print(f"{id} --> {confid}")
    cv2.imshow("frame", gray)
    k = cv2.waitKey(0) & 0xFF  # Press 'ESC' for exiting video

    # imagen.show(f"{id} --> {confid}")


def recognize_webcam():
    LBPHFaceRecognizer = get_LBPHFaceRecognizer()
    ids, names = get_names_ids()

    cam = cv2.VideoCapture(0)
    cam.set(3, W_FRAME)  # set Width
    cam.set(4, H_FRAME)  # set Height

    while True:
        ret, frm = cam.read()
        frame_gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        face = detect_face(frame_gray)

        if not face:
            cv2.imshow("frame", frm)
            k = cv2.waitKey(10) & 0xFF  # Press 'ESC' for exiting video
            continue

        for (x, y, xx, yy) in face:
            img = np.array(frame_gray[y:yy, x:xx], "uint8")
            id, confid = LBPHFaceRecognizer.predict(img)

            if confid < 30:
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
