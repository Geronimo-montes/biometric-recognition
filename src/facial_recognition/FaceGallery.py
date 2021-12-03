#!/usr/bin/env python3
# coding=utf-8

import os
import cv2

from utils.utils import PATH_DATABASE
from utils.utils import MIN_SIZE
from utils.utils import PATH_MODEL

from utils.utils import get_LBPHFaceRecognizer
from utils.utils import get_face_detector
from utils.utils import get_images_labels
from utils.utils import set_metod_input_data


def add_person(name: str, video: str = None, activeCamara: bool = True):
    """Se escanea el rotro de una persona nueva, se generan capturas de su rostro y se almacenan en un directorio con su nombre.

    Parameters
    ----------
    `name` : str
        Nombre de la persona que se esta agregando
    `video` : str, optional
        Nombre del video que se desea analizar. Si la opcion activeCamara esta en true este valor es ignorado aunque tenga un valor valido. Los videos a leer deben ubicarse en `PATH.ROOT.DATA.INPUT`, by default None
    `activeCamara` : bool, optional
        Si esta establecido en `True` indica que la entrada de datos es mediante la camara, para el caso de ser `False` se espera un video/img para analizar, by default True

    Raises
    ------
    InputMethodNotSelected
        Exepcion que se produce al no seleccionarse un metodo de entrada valido.
    """
    face_detector = get_face_detector()

    dir_name = f"{len(os.listdir(PATH_DATABASE))}_{name.upper()}"
    path_data = os.path.join(PATH_DATABASE, dir_name)

    if not os.path.exists(path=path_data):
        os.makedirs(path_data)

    cam = set_metod_input_data(activeCamara=activeCamara, path_video=video)

    count = 0
    while True:
        ret, frame = cam.read()
        # PROCESAMOS EL FRAME A ESCALA DE GRISES
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # DETECTAMOS EL ROSTRO
        faces = face_detector.detectMultiScale(
            frame_gray, scaleFactor=1.2, minNeighbors=5, minSize=MIN_SIZE
        )

        for (x, y, w, h) in faces:
            count += 1
            # RESALTAMOS EL RESTRO DETECTADO EN EL FRAME ORIGINAL
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # RECORTAMOS EL RESOTRO DEL FRAME A ESCALA DE GRISES
            img = frame_gray[y : y + h, x : x + w]
            # GUARDAMOS LA IMAGEN EN DISCO
            cv2.imwrite(os.path.join(path_data, f"{count}.jpg"), img)

        cv2.imshow("frame", frame)

        k = cv2.waitKey(50) & 0xFF  # Press 'ESC' for exiting video

        if k == 27 or count >= 100:
            break

    cam.release()
    cv2.destroyAllWindows()

    _faces, _ids = get_images_labels(
        face_detector=face_detector, path_data=PATH_DATABASE, name_folder=dir_name
    )

    train((_faces, _ids))


def train(database=None):
    """Escanea las capturas almaceadas y las prepara para entrenar el modelo. Al finalizar los almacena en `PATH.ROOT`."""

    LBPHFaceRecognizer = get_LBPHFaceRecognizer()
    face_detector = get_face_detector()

    if database == None:
        faces, ids = get_images_labels(
            face_detector=face_detector, path_data=PATH_DATABASE
        )
    else:
        faces, ids = database

    LBPHFaceRecognizer.update(faces, ids)
    LBPHFaceRecognizer.write(PATH_MODEL)


def recognize(video: str = None, activeCamara: bool = True):
    """Se indicada el metodo de entrada es mediante la captura de video o si se trata de un video almacenado. Se analizan los frames, si detecta rostros intentara clasificarlos

    Parameters
    ----------
    video : str, optional
        Nombre del video que se desea analizar. Si la opcion activeCamara esta en true este valor es ignorado aunque tenga un valor valido. Los videos a leer deben ubicarse en `PATH.ROOT.DATA.INPUT`, by default None
    activeCamara : bool, optional
        Si esta establecido en `True` indica que la entrada de datos es mediante la camara, para el caso de ser `False` se espera un video/img para analizar, by default True

    Raises
    ------
    InputMethodNotSelected
        Exepcion que se produce al no seleccionarse un metodo de entrada valido.
    """

    # CAPTURAMOS EL MODELO ALMACENADO
    LBPHFaceRecognizer = get_LBPHFaceRecognizer()
    face_detector = get_face_detector()

    ids, names = [], []

    for _dir in os.listdir(PATH_DATABASE):
        _a, _b = _dir.split("_")
        ids.append(int(_a))
        names.append(_b)

    cam = set_metod_input_data(activeCamara=activeCamara, path_video=video)

    while True:
        ret, frm = cam.read()
        # PROCESAMOS EL FRAME A ESCALA DE GRISES
        frame_gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        # DETECTAMOS EL ROSTRO DENTRO DEL FRAME
        face = face_detector.detectMultiScale(
            frame_gray, scaleFactor=1.2, minNeighbors=5, minSize=MIN_SIZE
        )

        # RECORTAMOS EL ROSTRO DETECTADO DEL FRAME ORIGNAL
        for (x, y, w, h) in face:
            img = frame_gray[y : y + h, x : x + w]
            # APLICAMOS EL ALGORITMO DE RECONOSIMIENTO
            id, confid = LBPHFaceRecognizer.predict(img)

            # If confidence is less them 100 ==> "0" : perfect match
            if confid < 60:
                index = ids.index(id)
                id = names[index]
                color = (0, 255, 0)  # verde
            else:
                id = "unknown"
                color = (0, 0, 255)  # rojo

            confid = "  {:3.1f}%".format(round(100 - confid))

            cv2.rectangle(frm, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frm, str(id), (x + 5, y - 5), 2, 1, color, 4)
            cv2.putText(frm, str(confid), (x + 5, y + h - 5), 2, 1, color, 2)

        cv2.imshow("frame", frm)

        k = cv2.waitKey(10) & 0xFF  # Press 'ESC' for exiting video
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
