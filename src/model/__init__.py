"""
:mod:`facial_reconnition`: Modulo de reconosimiento biometrico de rostros.
"""
from .train import train
from .add import add_webcam
from .add import add_galery
from .add import add_galery_to_db
from .recognize import detect_face, recognize_webcam, recognize_image, recognize_video

__all__ = [
    "train",
    #
    "add_webcam",
    "add_galery",
    "add_galery_to_db",
    #
    "recognize_webcam",
    "recognize_image",
    "recognize_video",
    #
    "detect_face",
]
