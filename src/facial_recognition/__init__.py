"""
:mod:`facial_reconnition`: Modulo de reconosimiento biometrico de rostros.
"""
from .FaceGallery import add_person, train, recognize
from .FaceGalleryError import InputMethodNotSelected

__all__ = [
    "add_person",
    "train",
    "recognize",
    "InputMethodNotSelected",
]
