"""
:mod:`facial_reconnition`: Modulo de reconosimiento biometrico de rostros.
"""
from .train import train
from .face_recognize import detect_face
from .add import add_from_webcam
from .add import add_from_galery
from .add import add_from_galery_direct_to_database
from .recognize import recognize_webcam
from .recognize import recognize_image
from .recognize import recognize_video

__all__ = [
    "train",
    #
    "add_from_webcam",
    "add_from_galery",
    "add_from_galery_direct_to_database",
    #
    "recognize_webcam",
    "recognize_image",
    "recognize_video",
    #
    "detect_face",
]
