import json
import os
import shutil
import sys
from enum import Enum
from typing import List

from utils.database import set_data_train
from utils.utils import PATH_ROOT, set_new_id
from utils.utils import PATH_DATA
from utils.utils import PATH_MODEL
from utils.utils import save_names_list

from face_recognize.add import add_from_galery
from face_recognize.add import add_from_webcam

from face_recognize.train import train

from face_recognize.recognize import recognize_image
from face_recognize.recognize import recognize_webcam

PATH_DIR_PRIVATE = os.path.join(
    PATH_ROOT, "..", "..", "api-biometric-recognition", "src", "assets", "private"
)
PATH_DIR_TRAIN = os.path.join(PATH_DATA, "train")
PATH_DIR_TEST = os.path.join(PATH_DATA, "test")


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
    def run(args: List[str]):
        print("Programa ejecutandose...", flush=True, end="")
        # sys.stdout.flush()
        ###########################################
        if EAccion.ADD_WEBCAM.value in args[0]:
            print("Add Video To Model...", flush=True, end="")
            add_from_webcam(name="geronimo")

        ###########################################
        if EAccion.ADD_GALERY.value in args[0]:
            if os.path.exists(PATH_MODEL):
                os.remove(PATH_MODEL)

            # save_names_list(["briss", "carlos", "chavez", "david", "geronimo"])
            # train()

            if not args[0]:
                raise Exception("Argumento name obligatorio...")

            print("Add Images To Model...", flush=True, end="")
            # sys.stdout.flush()
            name = args[1]

            if os.path.exists(PATH_DIR_TRAIN):
                if len(os.listdir(PATH_DIR_TRAIN)) > 0:
                    shutil.rmtree(PATH_DIR_TRAIN)
                    print("Dir train delete...", flush=True, end="")
                    # sys.stdout.flush()

            os.makedirs(os.path.join(PATH_DIR_TRAIN, name))
            print("Dir train create...", flush=True, end="")
            # sys.stdout.flush()

            if os.path.exists(PATH_DIR_PRIVATE):
                files = os.listdir(PATH_DIR_PRIVATE)
                files.remove("index.ts")
                # sys.stdout.flush()

                for index in range(len(files)):
                    src = os.path.join(PATH_DIR_PRIVATE, files[index])
                    dst = os.path.join(PATH_DIR_TRAIN, name, f"{name}_{files[index]}")
                    shutil.move(src, dst)

            for _d in os.listdir(os.path.join(PATH_DATA, "train")):
                name = _d.replace("_", "-")
                print(f"Analizando rostro: {_d}...", flush=True, end="")
                # sys.stdout.flush()

                add_from_galery(name, os.path.join(PATH_DIR_TRAIN, _d))
                print(f"Rostro agregado: {_d}...", flush=True, end="")
                # sys.stdout.flush()
                # os.path.join(os.pa        th.join(PATH_DATA, "train")

            print(f"Proceso terminado...", flush=True, end="")
            # sys.stdout.flush()

            if os.path.exists(PATH_DIR_TRAIN):
                if len(os.listdir(PATH_DIR_TRAIN)) > 0:
                    shutil.rmtree(PATH_DIR_TRAIN)
                    print("Dir train delete...", flush=True, end="")
                    # sys.stdout.flush()

        ###########################################
        if EAccion.RECOGNIZE_WEBCAM.value in args[0]:
            print("Recognize Video...", flush=True, end="")
            recognize_webcam()

        ###########################################
        if EAccion.RECOGNIZE_GALERY.value in args[0]:
            print("Recognize Galery...", flush=True, end="")
            path_test = os.path.join(PATH_DATA, "test")

            res = []
            for img in os.listdir(path_test):
                print(f"Analizando Imagen... {img}", flush=True, end="")
                res.append(recognize_image(os.path.join(path_test, img)))

            with open("resultados.json", "w") as f:
                json.dump({"list": res}, f)

        ###########################################
        if EAccion.TRAIN_MODEL.value in args[0]:
            print("Train Model...", flush=True, end="")
            train()

        ###########################################
        # if EAccion.PREPARE_DATASET.value in args[0]:
        #     print("Prerando Dataset", flush=True, end="")
        #     set_data_train()


if __name__ == "__main__":
    args = sys.argv[1:]
    Main.run(args)
