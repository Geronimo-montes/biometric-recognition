<p align="center">
  <a href="" rel="noopener">
 <img src="./docs/banner.png" alt="Project logo"></a>
</p>

# Facial Recognition
<div align="center">

  [![](https://img.shields.io/badge/status-active-success.svg)]()
  [![](https://img.shields.io/badge/category-machine_learning-red.svg)]()

</div>

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Contributing](../CONTRIBUTING.md)

## About <a name = "about"></a>

Projecto de reconosimiento facial utilizando python.

## Getting Started <a name = "getting_started"></a>

### Prerequisites

- Python 3.9
- Lib ' virtualenv '

### Installing

```bash
# CREATE DATA FOLDER PARA LA COMUNICACÓN ENTRE NODEJS Y PYTHON
$ mkdir ../data

# CREATE VIRTUAL ENV IN CURRENT FOLDER
$ python -m venv .venv

# INSTALL REQUIREMENTS
$ pip install -r requirements.txt
```

## Usage <a name = "usage"></a>

Lista de commandos 

``` bash
# EXAMINA UN VIDEO TOMADOS DE LA UBUICACION `../DATA` Y AGREGA LOS ROSTROS DETECTADOS PARA DESPUES ENTRENAR EL MODELO
$ python src/main.py --add_webcam --name {{nombre_persona}}

# EXAMINA UN CONJUNTO DE IMAGENES TOMADOS DE LA UBUICACION `../DATA` Y AGREGA LOS ROSTROS DETECTADOS PARA DESPUES ENTRENAR EL MODELO
$ python src/main.py --add_galery --name {{nombre_persona}}

# EXAMINA UN VIDEO TOMADOS DE LA UBUICACION `../DATA` Y DETERMINA SI ES UN ROSTRO CONOSIDO O DESCONOSIDO
$ python src/main.py --recognize_webcam

# EXAMINA UN CONJUNTO DE IMAGENES TOMADOS DE LA UBUICACION `../DATA` Y DETERMINA SI ES UN ROSTRO CONOSIDO O DESCONOSIDO
$ python src/main.py --recognize_galery

# GENERA EL MODELO ENTRENADO CON LOS DATOS DISPONIBLES EN EL DIRECTORIO `MODEL/DATABASE/`
$ python src/main.py --train
```
