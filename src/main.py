import json
import os

from utils.argumentos import load_args

from utils.settings import PATH_DATA_MODEL

from model.add import add_galery, add_galery_to_db
from model.add import add_webcam

from model.train import train

from model.recognize import recognize_image
from model.recognize import recognize_webcam

if __name__ == "__main__":
    args = load_args()
    print("Run Face Model...", flush=True, end="\t")

    #######################################################################
    if args.add_webcam:
        if not args.name:
            raise Exception("No Provider Name")
            
        add_webcam(name=args.name)
    #######################################################################
    elif args.add_galery:
        if not args.name:
            raise Exception("No Provider Name")

        print(f"Analizando galeria: {args.name}...", flush=True, end="")
        name, count, id = add_galery(args.name, PATH_DATA_MODEL)

        with open("resultado.json", "w") as f:
            json.dump({"name": name, "num_files": count, "id": id}, f)
    #######################################################################
    elif args.add_galery_to_db:
        if not args.name:
            raise Exception("No Provider Name")

        print(f"Analizando galeria: {args.name}...", flush=True, end="")

        name, count, id = add_galery_to_db(args.name, PATH_DATA_MODEL)

        with open("resultado.json", "w") as f:
            json.dump({"name": name, "num_files": count, "index": id}, f)
    #######################################################################
    elif args.recognize_webcam:
        recognize_webcam()
    #######################################################################
    elif args.recognize_galery:
        print("Run Recgnize Face From Galery...", flush=True, end="\t")
        names, prom_confid = [], []
        for img in os.listdir(PATH_DATA_MODEL):
            print(f"Analizando Imagen... {img}", flush=True, end="")
            id, confid = recognize_image(os.path.join(PATH_DATA_MODEL, img))
            if id != "unknown":
                names.append(id)
                prom_confid.append(confid)

        if len(names) == 0:
            print("unknown", flush=True, end="")
        elif len(names) != names.count(names[0]):
            print("unknown", flush=True, end="")
        else:
            print(names[0], flush=True, end="")
    #######################################################################
    elif args.train:
        train()
    #######################################################################

    # # CLEAN DIR PATH_DATA
    # for f in os.listdir(PATH_DATA_MODEL):
    #     os.remove(os.path.join(PATH_DATA_MODEL, f))
