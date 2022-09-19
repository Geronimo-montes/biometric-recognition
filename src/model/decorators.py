"""
Pruebas de implementacion de decoradores en codigo Pyhon.
Fecha:
    08-09-2022
Descripcion:
    Decorador que recibe una lista de parametros que valida existan en la funcion donde se implementa. Ademas valida la existencia del directorio donde se almacenan los datos para el modelo de reconosimiento.
"""
import os
from typing import List
from model.train import train
from utils.settings import PATH_DATABASE, save_names_list


def validator_method_add(params_name: List = [], *args, **kwargs):
    def execute_method(func):
        def wrapper(*args, **kwargs):
            # ADD VALIDACIONES DE PARA METROS
            print(args, kwargs)

            for param_name in params_name:
                if not param_name in kwargs:
                    raise Exception(f"NOT PROVIDER PARAM {param_name.upper()}")

            imgs = os.listdir(PATH_DATABASE)
            if len(imgs) < 1:
                save_names_list([])

            resul = func(*args, **kwargs)
            # TODO: IMPLEMENTAR LLAMDO A FUCION DE ENTRENAMIENTO
            # train()
            return resul

        return wrapper

    return execute_method
