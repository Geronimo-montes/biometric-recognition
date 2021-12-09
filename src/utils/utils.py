import os
import json
import cv2
import numpy as np

from typing import List
from PIL import Image
from os.path import abspath
from os.path import dirname
from os.path import join

from cv2.data import haarcascades
from cv2 import CascadeClassifier


PATH_ROOT = abspath(join(dirname(__file__), "..", ".."))
"""`PATH.ROOT`: Ruta absoluta de la ubicacion del proyecto."""

PATH_DATABASE = abspath(join(dirname(__file__), "..", "..", "data", "database"))
"""`PATH.ROOT.DATA.DATASET`: Ruta absoluta de la ubicacion de la carpeta data."""

PATH_INPUT = abspath(join(dirname(__file__), "..", "..", "data", "input"))
"""`PATH.ROOT.DATA.INPUT`: Ruta absoluta de la ubicacion de la entrada de datos."""

PATH_MODEL = abspath(join(dirname(__file__), "..", "..", "model", "modeloLBPHFace.xml"))
"""`PATH.ROOT.MODEL.MODEL`: Ruta absoluta de la ubicacion del modelo entrenado."""

PATH_NAMES = abspath(join(dirname(__file__), "..", "..", "model", "names.json"))
"""`PATH.ROOT.MODEL.NAMES`: Ruta absoluta de la ubicacion del archivo donde se almacenan los nombres."""

W_FRAME = 1024
# W_FRAME = 640
"""`FRAME.WIDTH` Dimensiones del la panatalla a mostrar."""

H_FRAME = 728
# H_FRAME = 480
"""`FRAME.HEIGHT` Dimensiones del la panatalla a mostrar."""

FACTOR = 0.3
""""""

MIN_SIZE = (int(W_FRAME * FACTOR), int(H_FRAME * FACTOR))
"""`FRAME.MIN_SIZE` Dimensiones minimas para el area de detección de rostros."""


def get_LBPHFaceRecognizer():
    LBPHFaceRecognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=30
    )

    if os.path.exists(path=PATH_MODEL):
        LBPHFaceRecognizer.read(PATH_MODEL)

    return LBPHFaceRecognizer


def get_face_detector():
    face1 = "haarcascade_frontalface_alt.xml"
    face2 = "haarcascade_frontalface_alt2.xml"
    face3 = "haarcascade_frontalface_alt_tree.xml"
    face4 = "haarcascade_frontalface_default.xml "

    return CascadeClassifier(haarcascades + face4)


def get_eye_detector():
    cascade_name1 = "haarcascade_eye_tree_eyeglasses.xml"
    cascade_name2 = "haarcascade_eye.xml"

    return CascadeClassifier(haarcascades + cascade_name1)


def get_images_labels():
    faces, ids = [], []

    for img in os.listdir(PATH_DATABASE):
        id, name, count = img.split("_")
        PIL_img = Image.open(os.path.join(PATH_DATABASE, img))
        img_numpy = np.array(PIL_img, "uint8")

        faces.append(img_numpy)
        ids.append(int(id))

    return faces, np.array(ids)


def load_names_list():
    with open(PATH_NAMES, "r") as f:
        data = json.load(f)
    return data["list"]


def save_names_list(names: List[str]):
    with open(PATH_NAMES, "w") as f:
        json.dump({"list": names}, f)


def set_new_id(name: str):
    data = load_names_list()
    id = len(data)

    data.append(name)
    save_names_list(data)

    return id


def get_names_ids():
    names = load_names_list()
    ids = [*range(0, len(names), 1)]

    return ids, names
