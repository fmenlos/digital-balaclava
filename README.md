Research software to avoid one's face from being recognized.
Developed for UNIR's "Master Universitario en Inteligencia Artificial".

Create some images for training in the "detector/training" folder (png format).
Train the face detector script with "./face_detector.py --known_people_folder  training/"
Create as many "config.ini.99" files as you need. This configuration file allows to fine-tune tests.
Use "runTests.sh 99" to generate simulated images and pass them through the face_detector. 
Some filtering is done with thje 