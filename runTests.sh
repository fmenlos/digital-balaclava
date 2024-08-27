#!/bin/sh
cp config.ini.$1 config.ini
./simulated_camera_publisher.py &
./processor_subscriber_publisher.py &
sleep 3
p_uno=$(pgrep -f simulated_camera_publisher.py)
p_dos=$(pgrep -f processor_subscriber_publisher.py)

mkdir humanos_test
./simulated_display_subscriber.py
mv humanos_test detector
mv detector/humanos_test detector/validation$1
cd detector
./face_detector.py validation$1 > validation$1.csv
cd ..
kill $p_uno
kill $p_dos