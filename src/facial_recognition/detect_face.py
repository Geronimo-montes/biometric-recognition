#!/usr/bin/env python3
# coding=utf-8
from typing import List, Tuple
from utils.utils import MIN_SIZE
from utils.utils import get_eye_detector, get_face_detector


def detect_face(imagen) -> List[Tuple]:
    """Detecta el rostro de una presona en una imagen mediante los metodos de deteccion de ojos y rotro incluidos en la libreria opencv.

    Parameters
    ----------
    imagen : Image
        Imagen en donde se desea detectar el rotros

    Returns
    -------
    Tuple
        Tupla con las cordenadas del rectangulo correspondiente al rotros dentro de la imangen. (x, y, xx, yy)
    """
    face_detector = get_face_detector()
    eyes_detector = get_eye_detector()

    faces = face_detector.detectMultiScale(
        imagen, scaleFactor=1.3, minNeighbors=5, minSize=MIN_SIZE
    )

    if len(faces) == 0:
        return None

    for (fx, fy, fw, fh) in faces:
        face = imagen[fy : fy + fh, fx : fx + fw]
        eyes = eyes_detector.detectMultiScale(face, scaleFactor=1.3, minNeighbors=5)

        if len(eyes) != 2:
            return None

        (a, b, c, d) = eyes[0]
        (aa, bb, cc, dd) = eyes[1]

        x = fx + (a if a < aa else aa)
        y = fy + (b if b < bb else bb)
        xx = fx + (a + c if a > aa else aa + cc)
        yy = fy + fh

        return [(x, y, xx, yy)]
