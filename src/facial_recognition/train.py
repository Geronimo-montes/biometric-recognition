#!/usr/bin/env python3
# coding=utf-8

import os
from utils.utils import PATH_MODEL

from utils.utils import get_LBPHFaceRecognizer
from utils.utils import get_images_labels


def train(database=None):
    """Escanea las capturas almaceadas y las prepara para entrenar el modelo. Al finalizar los almacena en `PATH.ROOT`."""

    if database == None:
        if os.path.exists(PATH_MODEL):
            os.remove(PATH_MODEL)
        faces, ids = get_images_labels()
    else:
        faces, ids = database

    LBPHFaceRecognizer = get_LBPHFaceRecognizer()
    LBPHFaceRecognizer.update(faces, ids)
    LBPHFaceRecognizer.write(PATH_MODEL)
