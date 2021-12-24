import argparse


def load_args():
    """Carga los argumentos soportados por el scrip.

    Args
    ----
    `[ -aw, --add_webcam ]`:  Add person from webcam

    `[ -ag, --add_galery ]`:  Add person from galery

    `[ -rw, --recognize_webcam ]`:  Recognoze person from webcam

    `[ -rg, --recognize_galery ]`:  Recognoze person from galery

    `[ -t, --train ]`:  Train model

    `[ -n, --name ]`:  Name of person

    """

    parse = argparse.ArgumentParser()

    # ADD METODS
    parse.add_argument(
        "-aw",
        "--add_webcam",
        help="Add person from webcam",
        action="store_true",
    )
    parse.add_argument(
        "-ag",
        "--add_galery",
        help="Add person from galery",
        action="store_true",
    )
    # RECOGNIZE METODS
    parse.add_argument(
        "-rw",
        "--recognize_webcam",
        help="recognoze person from webcam",
        action="store_true",
    )
    parse.add_argument(
        "-rg",
        "--recognize_galery",
        help="recognoze person from galery",
        action="store_true",
    )
    # TRAIN MODEL
    parse.add_argument(
        "-t",
        "--train",
        help="Train model",
        action="store_true",
    )
    # OTHERS PARAMS
    parse.add_argument(
        "-n",
        "--name",
        help="Name of person",
    )

    return parse.parse_args()
