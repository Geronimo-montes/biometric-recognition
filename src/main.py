import os

from utils.argumentos import load_args

from utils.utils import PATH_DATA

from face_recognize.add import add_from_galery
from face_recognize.add import add_from_webcam

from face_recognize.train import train

from face_recognize.recognize import recognize_image
from face_recognize.recognize import recognize_webcam

TEMP = os.path.join(PATH_DATA, "temp")

if __name__ == "__main__":
    args = load_args()
    print("Run Face Model...", flush=True, end="\t")

    if args.add_webcam:
        if not args.name:
            raise Exception("No Provider Name")
        add_from_webcam(name=args.name)

    elif args.add_galery:
        if not args.name:
            raise Exception("No Provider Name")
        print(f"Analizando galeria: {args.name}...", flush=True, end="")
        add_from_galery(args.name, TEMP)

    elif args.recognize_webcam:
        recognize_webcam()

    elif args.recognize_galery:
        print("Run Recgnize Face From Galery...", flush=True, end="\t")
        names, prom_confid = [], []
        for img in os.listdir(TEMP):
            print(f"Analizando Imagen... {img}", flush=True, end="")
            id, confid = recognize_image(os.path.join(TEMP, img))
            if id != "unknown":
                names.append(id)
                prom_confid.append(confid)

        if len(names) == 0:
            print("unknown", flush=True, end="")
        elif len(names) != names.count(names[0]):
            print("unknown", flush=True, end="")
        else:
            print(names[0], flush=True, end="")

    elif args.train:
        train()

    # CLEAN DIR TEMP
    for f in os.listdir(TEMP):
        os.remove(os.path.join(TEMP, f))
