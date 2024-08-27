Research software to avoid one's face from being recognized.
Developed for UNIR's "Master Universitario en Inteligencia Artificial".

1. Create some images for training in the "detector/training" folder (png format).
2. Train the face detector script with "./face_detector.py --known_people_folder  training/"
3. Create as many "config.ini.99" files as you need. This configuration file allows to fine-tune tests.
4. Use "runTests.sh 99" to generate simulated images and pass them through the face_detector. 
5. Some filtering is done with the "readyTests.sh 99" script