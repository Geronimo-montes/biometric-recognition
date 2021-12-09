import os
import sys
from enum import Enum
from utils.utils import PATH_INPUT

from facial_recognition.add import add_from_galery
from facial_recognition.add import add_from_webcam

from facial_recognition.train import train

from facial_recognition.recognize import (
    recognize_image,
    recognize_video,
    recognize_webcam,
)


class EAccion(Enum):
    ADD_WEBCAM = "add_webcam"
    ADD_GALERY = "add_galery"
    #
    TRAIN_MODEL = "train"
    #
    RECOGNIZE_WEBCAM = "run_webcam"
    RECOGNIZE_GALERY = "run_galery"


class Main:
    @staticmethod
    def run(*args, **dict):
        print("Programa ejecutandose...")

        if EAccion.ADD_WEBCAM.value in args[0]:
            add_from_webcam(name="geronimo")

        if EAccion.ADD_GALERY.value in args[0]:
            path = os.path.join(PATH_INPUT, "train")
            for _dir in os.listdir(path):
                print(f"ANALIZANDO:... {_dir}")
                id, name = _dir.split("_")
                add_from_galery(name, os.path.join(path, _dir))

        if EAccion.RECOGNIZE_WEBCAM.value in args[0]:
            recognize_webcam()

        if EAccion.RECOGNIZE_GALERY.value in args[0]:
            path_test = os.path.join(PATH_INPUT, "test")
            for img in os.listdir(path_test):
                print(f"Analizando Imagen... {img}")
                recognize_image(os.path.join(path_test, img))

        if EAccion.TRAIN_MODEL.value in args[0]:
            train()


if __name__ == "__main__":
    # args = sys.argv[1:]
    # Main.run(args)
    recognize_video()
