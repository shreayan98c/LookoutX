import cv2  # OpenCV library
import time
import requests  # python http library
# import pyaudio  # python audio library
import pyttsx3  # python text to speech library
# import speech_recognition as sr  # python speech recognition library
import numpy as np  # python numerical library
# import matplotlib.pyplot as plt


def stream(wifi_ip, port):

    livestream_url = f'http://{wifi_ip}:{port}/video'

    # Initializing video
    video = cv2.VideoCapture(livestream_url)

    # Initialize the flag variable
    flag = True

    while True:
        check, frame = video.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        if flag:
            test_frame = frame
            flag = False

        # Waiting for 1ms
        key = cv2.waitKey(1)
        # Break if user presses quit
        cv2.imshow('Capturing', frame)

        # plt.show()
        # plt.imshow(cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB))
        # plt.show()

        if key == ord('q'):
            break

    # Release the VideoCapture object
    video.release()

    cv2.destroyAllWindows()  # Destroy all windows
