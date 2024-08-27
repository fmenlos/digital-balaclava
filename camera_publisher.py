#!/usr/bin/env python
import configparser
from time import sleep

import cv2 as cv
import zmq

config = configparser.ConfigParser()
config.read("config.ini")

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.set_hwm(int(config["camera_publisher"]["high_water"]))
socket.bind(config["camera_publisher"]["bind"])

repeticiones = int(config["camera_publisher"]["repeticiones"])

TOPIC = "camera_frame"

try:
    cap = cv.VideoCapture(0)

    while cap.isOpened() and repeticiones != 0:
        ret, frame = cap.read()

        # se pudo captura otra imagen
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # pasamos a gris
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # publicamos imagen (multipart), si los receptores no son capaces
        socket.send_string(TOPIC, zmq.SNDMORE)
        socket.send_pyobj(frame)
        sleep(float(config["camera_publisher"]["retardo_entre_capturas"]))

        if repeticiones > 0:
            print(repeticiones)
            repeticiones = repeticiones - 1

except KeyboardInterrupt:
    print("Parando publicador...")
finally:
    # liberar recursos (zmq se cierra el solo)
    cap.release()
    print("Publicador PARADO!")
