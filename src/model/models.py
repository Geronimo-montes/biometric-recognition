import os
import cv2

from utils.settings import PATH_MODEL

THRESHOLD = 30


def get_LBPHFaceRecognizer():
    """Construye el modelo utilizado para el reconosimiento facial, utilizando el algoritmo LBPHF. Si existe un modelo en la ruta predeterminada lo carga al modelo construido."""

    Model_LBPH = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=THRESHOLD
    )

    if os.path.exists(path=PATH_MODEL):
        Model_LBPH.read(PATH_MODEL)

    return Model_LBPH


def get_face_detector():
    """Clasificador para detectar un rostro humano de frente."""
    face1 = "haarcascade_frontalface_alt.xml"
    face2 = "haarcascade_frontalface_alt2.xml"
    face3 = "haarcascade_frontalface_alt_tree.xml"
    face4 = "haarcascade_frontalface_default.xml "

    return cv2.CascadeClassifier(cv2.data.haarcascades + face1)


def get_eye_detector():
    """Clasificador que se encarga de la deteccion de ojos humanos."""
    cascade_name1 = "haarcascade_eye_tree_eyeglasses.xml"
    cascade_name2 = "haarcascade_eye.xml"

    return cv2.CascadeClassifier(cv2.data.haarcascades + cascade_name1)
