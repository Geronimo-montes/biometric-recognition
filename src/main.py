import argparse
import json
import os
import shutil
import sys
from enum import Enum
from typing import List
from utils.argumentos import load_args


from utils.utils import PATH_ROOT
from utils.utils import PATH_DATA
from utils.utils import PATH_MODEL

from face_recognize.add import add_from_galery
from face_recognize.add import add_from_webcam

from face_recognize.train import train

from face_recognize.recognize import recognize_image
from face_recognize.recognize import recognize_webcam

PATH_DIR_PRIVATE = os.path.join(
    PATH_ROOT, "..", "..", "api-biometric-recognition", "src", "assets", "private"
)
PATH_DIR_TEMP = os.path.join(PATH_DATA, "temp")


def prepare_data(name: str = ""):
    # GENERATE NEW MODEL
    if os.path.exists(PATH_MODEL):
        os.remove(PATH_MODEL)
    train()

    # DIR TEMP
    if os.path.exists(PATH_DIR_TEMP):
        shutil.rmtree(PATH_DIR_TEMP)
    os.makedirs(PATH_DIR_TEMP)
    # DIR SOURCE
    if os.path.exists(PATH_DIR_PRIVATE):
        print("Copy Dir Temp...", flush=True, end="\t")
        files = os.listdir(PATH_DIR_PRIVATE)
        files.remove("index.ts")
        name = "" if name == "" else name + "_"
        for index in range(len(files)):
            src = os.path.join(PATH_DIR_PRIVATE, files[index])
            dst = os.path.join(PATH_DIR_TEMP, f"{name}{files[index]}")
            shutil.move(src, dst)


if __name__ == "__main__":
    # args = sys.argv[1:]

    args = load_args()
    print("Run Face Model...", flush=True, end="\t")

    ######################################################################################
    if args.add_webcam:
        if not args.name:
            raise Exception("No Provider Name")

        add_from_webcam(name=args.name)

    ######################################################################################
    if args.add_galery:
        if not args.name:
            raise Exception("No Provider Name")

        prepare_data(args.name)
        print(f"Analizando galeria: {args.name}...", flush=True, end="")

        add_from_galery(args.name, PATH_DIR_TEMP)
        print(f"Proceso terminado...", flush=True, end="")

    ######################################################################################
    if args.recognize_webcam:
        recognize_webcam()

    ######################################################################################
    if args.recognize_galery:
        print("Run Recgnize Face From Galery...", flush=True, end="\t")
        prepare_data()

        names, prom_confid = [], []

        for img in os.listdir(PATH_DIR_TEMP):
            print(f"Analizando Imagen... {img}", flush=True, end="")

            id, confid = recognize_image(os.path.join(PATH_DIR_TEMP, img))

            if id != "unknown":
                names.append(id)
                prom_confid.append(confid)

        if len(names) == 0:
            print("unknown", flush=True, end="")
        elif len(names) != names.count(names[0]):
            print("unknown", flush=True, end="")
        else:
            print(names[0], flush=True, end="")
        # with open("resultados.json", "w") as f:
        #     json.dump(res, f)

    ######################################################################################
    if args.train:
        train()
