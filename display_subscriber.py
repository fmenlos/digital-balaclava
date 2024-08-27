#!/usr/bin/env python
import configparser

import cv2 as cv
import zmq

from utils import kmeans_color_quantization

config = configparser.ConfigParser()
config.read("config.ini")

TOPIC = b"display_frame"

context = zmq.Context()
socket_sub = context.socket(zmq.SUB)
socket_sub.connect(config["display_subscriber"]["sub"])
socket_sub.setsockopt(zmq.SUBSCRIBE, TOPIC)

try:
    while True:
        _ = socket_sub.recv_string()
        frame = socket_sub.recv_pyobj()

        frame = kmeans_color_quantization(frame,int(config["general.display"]["colores"]))

        cv.imshow("gray + blur", frame)
        cv.waitKey(1)

except KeyboardInterrupt:
    print("Parando display...")
finally:
    # liberar recursos (zmq se cierra el solo)
    print("Display PARADO!")
