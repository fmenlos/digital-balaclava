#!/usr/bin/env python
import configparser

from pathlib import Path

import cv2 as cv
import numpy as np
import zmq
from utils import kmeans_color_quantization


config = configparser.ConfigParser()
config.read("config.ini")


def salvar_imagen(nombre, imagen):
    status = cv.imwrite(
        nombre,
        imagen,
    )
    if not status:
        print(f"Error grabando imagen '{nombre}'!")


def cargar_imagen(nombre):
    return cv.imread(nombre, cv.IMREAD_UNCHANGED)


def escalar_rostro_con_distancia(entrada, dimensiones_salida, factor_distancia=1.0):
    veces_alto = (dimensiones_salida[0] / entrada.shape[0]) * factor_distancia
    veces_ancho = (dimensiones_salida[1] / entrada.shape[1]) * factor_distancia
    veces = np.min((veces_alto, veces_ancho))
    return cv.resize(
        entrada, (int(veces * entrada.shape[1]), int(veces * entrada.shape[0]))
    )


TOPIC_1 = b"camera_frame"
TOPIC_2 = b"display_frame"

context = zmq.Context()

# suscripciones a los 2 canales
socket_sub1 = context.socket(zmq.SUB)
socket_sub1.connect(config["simulated_display_subscriber"]["sub1"])
socket_sub1.setsockopt(zmq.SUBSCRIBE, TOPIC_1)
socket_sub2 = context.socket(zmq.SUB)
socket_sub2.connect(config["simulated_display_subscriber"]["sub2"])
socket_sub2.setsockopt(zmq.SUBSCRIBE, TOPIC_2)

repeticiones_por_humano = int(config["simulated_display_subscriber"]["repeticiones"])

folder_path = Path(config["simulated_display_subscriber"]["carpeta_origen"])

# relacion de aspecto entre una cara humana "estandar" y el tamaño del display
proporcion_display_cara_ancho = float(
    int(config["general.display"]["ancho_display"])
    / int(config["general.display"]["ancho_cara_media"])
)
proporcion_display_cara_alto = float(
    int(config["general.display"]["alto_display"])
    / int(config["general.display"]["alto_cara_media"])
)

offset_patron_x = float(config["simulated_display_subscriber"]["offset_patron_x"])
offset_patron_y = float(config["simulated_display_subscriber"]["offset_patron_y"])
salto_patron_y = float(config["simulated_display_subscriber"]["salto_patron_y"])


# motor de deteccion de caras, necesario para ajustar el tamaño del patron en proporcion a la cara que aparece en la imagen
faceCascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)
try:
    # recorre los archivos del carpeta de origen
    for file in folder_path.iterdir():

        # descartamos cualquier cosa que no sea un archivo PNG
        if file.is_dir() or file.is_file() and file.suffix != ".png":
            continue

        # cargamos el humano, separamos la cabeza y la convertimos a escala de grises
        cabeza_original = np.asarray(cv.imread(str(file), cv.IMREAD_UNCHANGED))[
            int(config["simulated_display_subscriber"]["inicio_zona_cabeza_y"]) : int(
                config["simulated_display_subscriber"]["final_zona_cabeza_y"]
            ),
            int(config["simulated_display_subscriber"]["inicio_zona_cabeza_x"]) : int(
                config["simulated_display_subscriber"]["final_zona_cabeza_x"]
            ),
        ]

        # obtenemos y difuminamos el fondo
        _ = socket_sub1.recv_string()
        fondo = cv.GaussianBlur(socket_sub1.recv_pyobj(), (75, 75), 0)

        # obtenermos el patron y lo redimensionamos a las proporciones de la cara
        _ = socket_sub2.recv_string()
        patron = socket_sub2.recv_pyobj()

        salvar_imagen(
            f"{config["simulated_display_subscriber"]["carpeta_destino"]}/{file.name}-patron{file.suffix}",
            patron,
        )

        factor_distancia = 1.0
        while factor_distancia > float(
            config["simulated_display_subscriber"]["factor_distancia_stop"]
        ):
            repeticiones = repeticiones_por_humano
            cabeza_ajustada = False

            while repeticiones != 0:

                if not cabeza_ajustada:
                    cabeza_ajustada = True

                    # amplia la cabeza hasta unas dimensiones acordes con el fondo
                    # se hace despues de obtener el mensaje con el fondo porque antes no tenemos las dimensiones
                    cabeza_original_escalada = escalar_rostro_con_distancia(
                        cabeza_original, fondo.shape, factor_distancia
                    )

                    # conservamos el canal de transparencia
                    alpha_channel = cabeza_original_escalada[:, :, 3] / 255

                    # pasa a escala de grises
                    cabeza_original_escalada = cv.cvtColor(
                        cabeza_original_escalada, cv.COLOR_BGR2GRAY
                    )

                    # busca una cara para ajustar las dimensiones del patron
                    _, _, cara_ancho, cara_alto = faceCascade.detectMultiScale(
                        cabeza_original_escalada,
                        scaleFactor=1.3,
                        minNeighbors=3,
                        minSize=(30, 30),
                    )[0]

                    # determina las dimensiones en pixels del patron en relación
                    # a las dimensiones físicas de display y de la cara
                    patron_ancho = int(cara_ancho * proporcion_display_cara_ancho)
                    patron_alto = int(cara_alto * proporcion_display_cara_alto)

                # ajusta los colores del patron a lo que soporta el display
                patron = kmeans_color_quantization(
                    patron, int(config["general.display"]["colores"])
                )

                patron = cv.resize(patron, (patron_ancho, patron_alto))

                # coloca la cabeza sobre el fondo capturado y suavizado
                # centrado en el ejex y pegado a la parte inferior
                h, w = cabeza_original_escalada.shape[:2]
                mitad_fondo = int(fondo.shape[1] / 2 - w / 2)
                mitad_fondo_mas_ancho_cabeza = mitad_fondo + w
                fragmento_fondo = fondo[
                    fondo.shape[0] - h : fondo.shape[0],
                    mitad_fondo:mitad_fondo_mas_ancho_cabeza,
                ]

                copia_fondo_antes_cabeza = np.copy(fragmento_fondo)

                cabeza_mas_fondo = (
                    fragmento_fondo * (1 - alpha_channel)
                    + cabeza_original_escalada * alpha_channel
                )

                # overwrite the section of the background image that has been updated
                fondo[
                    fondo.shape[0] - h : fondo.shape[0],
                    mitad_fondo:mitad_fondo_mas_ancho_cabeza,
                ] = cabeza_mas_fondo

                salvar_imagen(
                    f"{config["simulated_display_subscriber"]["carpeta_destino"]}/{file.name}-{factor_distancia:.2f}-cabeza_y_fondo{file.suffix}",
                    fondo,
                )

                # coloca el patron sobre el fondo y la cabeza
                pruebas_patron = 0
                while True:
                    h2, w2 = patron.shape[:2]

                    # calcula la posición del patron (posicion x esta definida por configuracion)
                    # posicion y se calcula en el lazo
                    pos_patron_y = int(
                        fondo.shape[0] * offset_patron_y
                        - pruebas_patron * salto_patron_y
                    )

                    # determina cuando finalizar el lazo de ubicaciones del patron
                    if pos_patron_y < 150 * (1 + (1 - factor_distancia) * 2.3):
                        break

                    pos_patron_x = int(fondo.shape[1] * offset_patron_x)

                    # conserva el contenido
                    copia_fondo_antes_patron = np.copy(
                        fondo[
                            pos_patron_y : pos_patron_y + h2,
                            pos_patron_x : pos_patron_x + w2,
                        ]
                    )
                    fondo[
                        pos_patron_y : pos_patron_y + h2,
                        pos_patron_x : pos_patron_x + w2,
                    ] = patron

                    salvar_imagen(
                        f"{config["simulated_display_subscriber"]["carpeta_destino"]}/{file.name}-{factor_distancia:.2f}-cabeza_y_fondo_y_patron{repeticiones*10 + pruebas_patron:02d}{file.suffix}",
                        fondo,
                    )

                    # recupera el contenido del fondo antes del reemplazo del patron
                    # de esta manera podemos utilizar la misma imagen para varias ubicaciones del patron
                    fondo[
                        pos_patron_y : pos_patron_y + h2,
                        pos_patron_x : pos_patron_x + w2,
                    ] = copia_fondo_antes_patron

                    pruebas_patron += 1

                fondo[
                    fondo.shape[0] - h : fondo.shape[0],
                    mitad_fondo:mitad_fondo_mas_ancho_cabeza,
                ] = copia_fondo_antes_cabeza

                if repeticiones > 0:
                    repeticiones = repeticiones - 1
            factor_distancia = factor_distancia - float(
                config["simulated_display_subscriber"]["factor_distancia_delta"]
            )


except KeyboardInterrupt:
    print("Parando display simulado...")
finally:
    # liberar recursos (zmq se cierra el solo)
    print("Display simulado PARADO!")
