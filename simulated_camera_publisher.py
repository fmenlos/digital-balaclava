#!/usr/bin/env python
import configparser

from pathlib import Path
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

folder_path = Path(config["simulated_camera_publisher"]["carpeta_origen"])

try:
    while repeticiones != 0:
        for file in folder_path.iterdir():
            if file.is_file() and file.suffix == ".png":
                # print(file)
                frame = cv.imread(str(file), cv.IMREAD_UNCHANGED)
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
    print("Publicador PARADO!")
