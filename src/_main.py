import os
import sys
import shutil

from utils.utils import PATH_ROOT
from utils.utils import PATH_DATA

PATH_DIR_PRIVATE = os.path.join(PATH_ROOT, "..", "..", "..", "assets", "private")
PATH_DIR_TRAIN = os.path.join(PATH_DATA, "train")
PATH_DIR_TEST = os.path.join(PATH_DATA, "test")


def add_images(name):
    if os.path.exists(PATH_DIR_TRAIN):
        if len(os.listdir(PATH_DIR_TRAIN)) > 0:
            shutil.rmtree(PATH_DIR_TRAIN)
            print("Dir train delete...")
            sys.stdout.flush()

    os.mkdirs(os.path.join(PATH_DIR_TRAIN, name))
    print("Dir train create...")
    sys.stdout.flush()

    if os.path.exists(PATH_DIR_PRIVATE):
        files = os.listdir(PATH_DIR_PRIVATE)
        files.remove("index.ts")

        for f in files:
            print(f)


if __name__ == "__main__":
    args = sys.argv[1:]
    add_images(args)
