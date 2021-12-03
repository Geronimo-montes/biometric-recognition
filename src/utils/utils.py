from os.path import join, abspath, dirname
import os
from sys import float_info as DBL_MAX
import numpy as np

from PIL import Image

import cv2
from cv2.data import haarcascades
from cv2 import CascadeClassifier
from cv2 import VideoCapture

from facial_recognition.FaceGalleryError import InputMethodNotSelected


PATH_ROOT = abspath(join(dirname(__file__), "..", ".."))
"""`PATH.ROOT`: Ruta absoluta de la ubicacion del proyecto."""

PATH_DATABASE = abspath(join(dirname(__file__), "..", "..", "data", "database"))
"""`PATH.ROOT.DATA:DATASET`: Ruta absoluta de la ubicacion de la carpeta data."""

PATH_INPUT = abspath(join(dirname(__file__), "..", "..", "data", "input"))
"""`PATH.ROOT.DATA.INPUT`: Ruta absoluta de la ubicacion de la carpeta input para la entrada de datos."""

PATH_MODEL = abspath(
    join(dirname(__file__), "..", "..", "data", "model", "modeloLBPHFace.xml")
)
"""`PATH.ROOT.DATA.MODEL`: Ruta absoluta de la ubicacion de la carpeta donde se almacena el modelo entrenado."""

W_FRAME = 640
"""`FRAME.WIDTH` Dimensiones del la panatalla a mostrar."""

H_FRAME = 480
"""`FRAME.HEIGHT` Dimensiones del la panatalla a mostrar."""

MIN_SIZE = (int(640 * 0.1), int(480 * 0.1))
"""`FRAME.MIN_SIZE` Dimensiones minimas para el area de detección de rostros."""

config_LBPHFaceRecognizer = [10, 8, 8, 8, DBL_MAX.max]


def get_LBPHFaceRecognizer():
    LBPHFaceRecognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=60
    )

    if os.path.exists(path=PATH_MODEL):
        LBPHFaceRecognizer.read(PATH_MODEL)

    return LBPHFaceRecognizer


def get_face_detector():
    cascade_name = "haarcascade_frontalface_default.xml"
    face_detector = CascadeClassifier(haarcascades + cascade_name)
    return face_detector


def set_metod_input_data(activeCamara: bool, path_video):
    # LEEMOS EL VIDEO DEL ROSTRO
    if activeCamara:
        cam = VideoCapture(0)
        cam.set(3, W_FRAME)  # set Width
        cam.set(4, H_FRAME)  # set Height
    elif path_video != None:
        cam = VideoCapture(os.path.join(PATH_INPUT, path_video))
    else:
        raise InputMethodNotSelected("Metodo de entrada de datos no seleccionado.")

    return cam


def get_images_labels(
    face_detector, path_data: str = PATH_DATABASE, name_folder: str = None
):
    if name_folder == None:
        paths_img = [os.path.join(path_data, p) for p in os.listdir(path_data)]
    else:
        paths_img = [os.path.join(path_data, name_folder)]

    faces = []
    ids = []

    for path_img in paths_img:
        id, name = os.path.basename(path_img).split("_")

        for img in os.listdir(path_img):
            _path = os.path.join(path_img, img)

            PIL_img = Image.open(_path).convert("L")  # convert it to grayscale
            img_numpy = np.array(PIL_img, "uint8")
            _faces = face_detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in _faces:
                faces.append(img_numpy[y : y + h, x : x + w])
                ids.append(int(id))

    return faces, np.array(ids)
