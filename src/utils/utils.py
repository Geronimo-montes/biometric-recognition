import os
import json
import numpy as np

from typing import List

from PIL import Image

from os.path import abspath
from os.path import dirname
from os.path import join


PATH_ROOT = abspath(join(dirname(__file__), "..", ".."))
"""`PATH.ROOT`: Ruta absoluta de la ubicacion del proyecto.
"""


PATH_DATA = abspath(join(dirname(__file__), "..", "..", "data"))
"""`PATH.ROOT.DATA.INPUT`: Ruta absoluta de la ubicacion de la entrada de datos.
"""


PATH_DATABASE = abspath(join(dirname(__file__), "..", "..", "model", "database"))
"""`PATH.ROOT.MODEL.DATASET`: Ruta absoluta de la ubicacion de la carpeta data.
"""


PATH_MODEL = abspath(join(dirname(__file__), "..", "..", "model", "modeloLBPHFace.xml"))
"""`PATH.ROOT.MODEL.MODEL`: Ruta absoluta de la ubicacion del modelo entrenado.
"""


PATH_NAMES = abspath(join(dirname(__file__), "..", "..", "model", "names.json"))
"""`PATH.ROOT.MODEL.NAMES`: Ruta absoluta del archivo donde se almacenan los nombres.
"""


W_FRAME = 1024  # W_FRAME = 640
"""`FRAME.WIDTH` Dimensiones del la panatalla a mostrar.
"""


H_FRAME = 768  # H_FRAME = 480
"""`FRAME.HEIGHT` Dimensiones del la panatalla a mostrar.
"""


FACTOR = 0.2
"""Factor de reduccion para calcular el tamaño minimo del recuadro de reconosimiento facial.
"""


# MIN_SIZE = (int(W_FRAME * FACTOR), int(H_FRAME * FACTOR))
MIN_SIZE = (100, 100)
"""`FRAME.MIN_SIZE` Dimensiones minimas para el area de detección de rostros.
"""


def get_train_data():
    """Genera el conjunto de datos para entrenar el modelo.

    Returns
    -------
    `Tuple[list, NDArray]` : `Tuple[faces, ids]`
        Conjunto de datos de entrenamiento para reconosimiento facial. La tupla contine la estructura
    """
    faces, ids = [], []

    for img in os.listdir(PATH_DATABASE):
        id, name, count = img.split("_")
        PIL_img = Image.open(os.path.join(PATH_DATABASE, img))
        img_numpy = np.array(PIL_img, "uint8")
        faces.append(img_numpy)
        ids.append(int(id))

    return (faces, np.array(ids))


def load_names_list():
    """Lista de nombres  ordenados deacuerdo al index asignado al dataset."""
    with open(PATH_NAMES, "r") as f:
        data = json.load(f)
    return data["list"]


def save_names_list(names: List[str]):
    """Guarda en memoria la lista de nombres  en formato json."""
    with open(PATH_NAMES, "w") as f:
        json.dump({"list": names}, f)


def set_new_id(name: str):
    """Calcula un nuevo id para un nuevo conjunto de rostros, agrega el nombre de la persona a la lista y finalmente guarda la lista en memoria."""
    data = load_names_list()
    id = len(data)
    # name  =name.replace("_", " ").capitalize()
    data.append(name)
    save_names_list(data)
    return id
