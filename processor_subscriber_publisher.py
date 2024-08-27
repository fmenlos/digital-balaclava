#!/usr/bin/env python
import configparser

import cv2 as cv
import numpy as np

import zmq

config = configparser.ConfigParser()
config.read("config.ini")


def salvar_archivo(salvar_archivos, nombre, frame):
    if salvar_archivos:
        status = cv.imwrite(f"./i/{nombre}.png", frame)
        if not status:
            print(f"Error grando archivo {nombre}!")


def cargar_patron(nombre,dimensiones_salida):
    imagen = cv.imread(nombre, cv.IMREAD_UNCHANGED)
    alpha_channel = imagen[:, :, 3] / 255
    alpha_channel = ajustar_patron(alpha_channel,dimensiones_salida)

    imagen = cv.cvtColor(imagen, cv.COLOR_BGRA2GRAY)
    imagen = ajustar_patron(imagen,dimensiones_salida)

    return imagen,alpha_channel


def ajustar_patron(entrada, dimensiones_salida):
    veces_alto = int(dimensiones_salida[0] / entrada.shape[0])
    veces_alto = veces_alto + 1
    veces_ancho = int(dimensiones_salida[1] / entrada.shape[1])
    veces_ancho = veces_ancho + 1
    return np.tile(np.asarray(entrada), (veces_alto, veces_ancho))[
        0 : dimensiones_salida[0], 0 : dimensiones_salida[1]
    ]


def resumir_imagen_fondo(entrada, dimensiones_salida=None):
    # si no nos especifican unas dimensiones tomamos las de la entrada
    if not dimensiones_salida:
        dimensiones_salida = entrada.shape

    indice_primer_tercio = int(entrada.shape[0] / 3)
    indice_segundo_tercio = indice_primer_tercio * 2

    media_tercio_superior = int(
        np.mean(entrada[0:indice_primer_tercio, 0 : entrada.shape[1]])
    )
    media_tercio_medio = int(
        np.mean(
            entrada[
                indice_primer_tercio + 1 : indice_segundo_tercio, 0 : entrada.shape[1]
            ]
        )
    )
    media_tercio_inferior = int(
        np.mean(entrada[indice_segundo_tercio + 1 :, 0 : entrada.shape[1]])
    )

    indice_primer_tercio_salida = int(dimensiones_salida[0] / 3)
    indice_segundo_tercio_salida = indice_primer_tercio_salida * 2

    salida = np.concat(
        (
            np.tile(
                np.repeat(media_tercio_superior, dimensiones_salida[1]),
                (indice_primer_tercio_salida, 1),
            ),
            np.tile(
                np.repeat(media_tercio_medio, dimensiones_salida[1]),
                (indice_segundo_tercio_salida - indice_primer_tercio_salida, 1),
            ),
            np.tile(
                np.repeat(media_tercio_inferior, dimensiones_salida[1]),
                (dimensiones_salida[0] - indice_segundo_tercio_salida, 1),
            ),
        ),
    )

    return salida.astype(np.uint8)


context = zmq.Context()
socket_sub = context.socket(zmq.SUB)
socket_sub.connect(config["processor"]["sub"])
socket_sub.setsockopt(zmq.SUBSCRIBE, b"camera_frame")


socket_pub = context.socket(zmq.PUB)
socket_pub.set_hwm(int(config["processor"]["high_water"]))
socket_pub.bind(config["processor"]["bind"])
TOPIC = "display_frame"

salvar_archivos = False

# dimensiones del display de salida
x = int(config["general.display"]["x"])
y = int(config["general.display"]["y"])

# carga el patron (que puede tener unas dimensiones diferentes a las del display de salida)
patron, alpha_patron = cargar_patron(config["processor"]["patron"],(y, x))
repeticiones = int(config["processor"]["repeticiones"])

try:
    while repeticiones != 0:
        _ = socket_sub.recv_string()
        original = socket_sub.recv_pyobj()
        salvar_archivo(salvar_archivos, "original", original)

        # obtener un "resumen" de la imagen de fondo
        imagen_resumida = resumir_imagen_fondo(original, (y, x))
        salvar_archivo(salvar_archivos, "resumido", imagen_resumida)
        salvar_archivo(salvar_archivos, "patron", patron)

        resultado = (
            imagen_resumida * (1 - alpha_patron) + patron * alpha_patron
        )

        salvar_archivo(salvar_archivos, "result", resultado)

        # reenvía la imagen original
        socket_pub.send_string(TOPIC, zmq.SNDMORE)
        # socket_pub.send_pyobj(original,zmq.SNDMORE)
        socket_pub.send_pyobj(resultado)

        if repeticiones > 0:
            print(repeticiones)
            repeticiones = repeticiones - 1


except KeyboardInterrupt:
    print("Parando procesador...")
finally:
    # liberar recursos (zmq se cierra el solo)
    print("Procesador PARADO!")
