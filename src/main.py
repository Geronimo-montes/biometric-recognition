import json
import os
import sys
from enum import Enum

from utils.database import set_data_train
from utils.utils import PATH_INPUT, PATH_MODEL, save_names_list

from face_recognize.add import add_from_galery
from face_recognize.add import add_from_webcam

from face_recognize.train import train

from face_recognize.recognize import (
    recognize_image,
    recognize_webcam,
)


class EAccion(Enum):
    #
    ADD_WEBCAM = "add_webcam"
    ADD_GALERY = "add_galery"
    #
    TRAIN_MODEL = "train"
    #
    RECOGNIZE_WEBCAM = "run_webcam"
    RECOGNIZE_GALERY = "run_galery"
    #
    PREPARE_DATASET = "prepare_dataset"


class Main:
    @staticmethod
    def run(*args, **dict):
        print("Programa ejecutandose...")

        ###########################################
        if EAccion.ADD_WEBCAM.value in args[0]:
            add_from_webcam(name="geronimo")

        ###########################################
        if EAccion.ADD_GALERY.value in args[0]:
            if os.path.exists(PATH_MODEL):
                os.remove(PATH_MODEL)

            save_names_list(["briss", "carlos", "chavez", "david", "geronimo"])
            train()

            for _d in os.listdir(os.path.join(PATH_INPUT, "train")):
                name = _d.replace("_", "-")
                print(f"ANALIZANDO:... {_d}")

                add_from_galery(
                    name, os.path.join(os.path.join(PATH_INPUT, "train"), _d)
                )

        ###########################################
        if EAccion.RECOGNIZE_WEBCAM.value in args[0]:
            recognize_webcam()

        ###########################################
        if EAccion.RECOGNIZE_GALERY.value in args[0]:
            path_test = os.path.join(PATH_INPUT, "test")

            res = []
            for img in os.listdir(path_test):
                print(f"Analizando Imagen... {img}")
                res.append(recognize_image(os.path.join(path_test, img)))

            with open("resultados.json", "w") as f:
                json.dump({"list": res}, f)

        ###########################################
        if EAccion.TRAIN_MODEL.value in args[0]:
            train()

        ###########################################
        if EAccion.PREPARE_DATASET.value in args[0]:
            print("RUN SELECT DATASET")
            set_data_train()


if __name__ == "__main__":
    args = sys.argv[1:]
    Main.run(args)
