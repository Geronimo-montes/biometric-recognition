import sys
from facial_recognition.FaceGallery import add_person, recognize, train
from enum import Enum

from facial_recognition.FaceGalleryError import InputMethodNotSelected


class EAccion(Enum):
    ADD_PERSON = "add"
    TRAIN_MODEL = "train"
    RECOGNIZE = "run"


class Main:
    @staticmethod
    def run(*args, **dict):
        print("Programa ejecutandose...")

        try:

            if EAccion.ADD_PERSON.value in args[0]:
                add_person(name="geronimo", video=None, activeCamara=True)

            if EAccion.RECOGNIZE.value in args[0]:
                recognize(video="1478_02_007_noam_chomsky.avi", activeCamara=True)

            if EAccion.TRAIN_MODEL.value in args[0]:
                train()

        except InputMethodNotSelected as err:
            print(err)


if __name__ == "__main__":
    args = sys.argv[1:]
    Main.run(args)
