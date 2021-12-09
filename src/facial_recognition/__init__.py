"""
:mod:`facial_reconnition`: Modulo de reconosimiento biometrico de rostros.
"""
from .add import add_from_webcam, add_from_galery
from .train import train
from .recognize import recognize_webcam, recognize_image, recognize_video
from .detect_face import detect_face

__all__ = [
    "add_from_webcam",
    "add_from_galery",
    "train",
    "recognize_webcam",
    "recognize_image",
    "recognize_video",
    "detect_face",
    "InputMethodNotSelected",
]
