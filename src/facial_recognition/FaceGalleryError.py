class Error(Exception):
    """Base clase error for exceptions in this module."""

    pass


class InputMethodNotSelected(Error):
    """Excepcion producida al no seleccionar un metodo de entrada de datos valida.

    Attributtes:
    ------------
    `expression` : str
        Expression de entrada o error producido.
    `message` : str
        Descripcion extendida de error.
    """

    def __init__(self, expression, menssaje):
        """Excepcion producida al no seleccionar un metodo de entrada de datos valida."""

        self.expression = expression
        self.menssaje = menssaje
