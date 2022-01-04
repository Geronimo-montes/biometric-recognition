import os
import cv2
import shutil
import random
import numpy as np
from PIL import Image

from face_recognize.face_recognize import detect_face

from utils.utils import PATH_DATA


PATH_DATABASE = os.path.join(os.path.dirname(__file__), "database")
DATA_TRAIN = os.path.join(PATH_DATA, "train")
DATA_TEST = os.path.join(PATH_DATA, "test")


def set_data_train():
    """ITERA LOS DATOS SELECCIONADOS PARA REMOVER LAS IMAGENES QUE NO TIENEN ROSTROS RECONOSIBLES AL MODELO, ADEMAS RENOMBRA LOS ARCHIVOS CON EL NOMBRE DE LA PERSONA Y UN INDECE"""

    # VACIAMOS LOS DIRECTORIOS TEST Y TRAIN
    for _d in [DATA_TEST, DATA_TRAIN]:
        shutil.rmtree(_d)
        os.mkdir(_d)

    for _dir in os.listdir(PATH_DATABASE):
        d_select = os.path.join(PATH_DATABASE, _dir)
        d_train = os.path.join(DATA_TRAIN, _dir)
        # COPIAMOS LA CARPETA ACTUAL AL DIRECTORIO DE DATOS DE PRUEBA
        shutil.copytree(d_select, d_train)

        # SELECCIONAMOS LAS IMAGENES VALIDAS
        for img in os.listdir(d_train):
            imagen = os.path.join(d_train, img)

            if not _isValid(imagen):
                os.remove(imagen)

        # SE NECESITAN AL MENOS 15 IMAGENES VALIDAS
        if len(os.listdir(d_train)) < 15:
            shutil.rmtree(d_train)
            continue

        name = _dir.replace("_", "-")

        # MANDAMOS 5 IMAGENES PARA PRUEBAS
        for i in range(5):
            imgs = os.listdir(d_train)
            index = random.randrange(1, len(imgs))
            img = os.path.join(d_train, imgs[index])
            shutil.move(img, os.path.join(DATA_TEST, f"{name}_{i}.jpg"))

        con = 0
        # RENOMBRAMOS LAS IMAGENES
        for _img in os.listdir(d_train):
            con += 1
            imagen = os.path.join(d_train, _img)

            if con > 10:
                os.remove(imagen)
                continue

            os.rename(imagen, os.path.join(d_train, f"{name}_{con}.jpg"))


def _isValid(img_path: str):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = detect_face(img)

    return False if not face else True
