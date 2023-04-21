import cv2  # OpenCV library
import time
import wave
import requests  # python http library
import pyaudio  # python audio library
import pyttsx3  # python text to speech library
# import speech_recognition as sr  # python speech recognition library
import numpy as np  # python numerical library
# import matplotlib.pyplot as plt


def stream(wifi_ip, port):

    livestream_url = f'http://{wifi_ip}:{port}/video'
    audio_url = f'http://{wifi_ip}:{port}/audio.wav'
    output_audio_file = 'audio_output.wav'

    # Initializing video
    video = cv2.VideoCapture(livestream_url)

    # Open the audio stream object
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

    # Open the output file for writing
    output_file = wave.open(output_audio_file, 'wb')
    output_file.setnchannels(1)
    output_file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    output_file.setframerate(44100)

    # Make a GET request to the DroidCam app to start the audio stream
    r = requests.get(audio_url, stream=True)

    # Initialize the flag variable
    flag = True

    while True:
        # Read a frame from the video stream
        check, frame = video.read()

        # Read a chunk of data from the audio stream
        audio_chunk = r.raw.read(1024)

        # Write the audio chunk to the output file
        output_file.writeframes(audio_chunk)

        # Convert the audio data to a numpy array and write it to the PyAudio stream
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        stream.write(audio_data)

        # Display the resulting frame
        if flag:
            test_frame = frame
            flag = False

        # Waiting for 1ms
        key = cv2.waitKey(1)
        cv2.imshow('Frame', frame)

        # Check for the 'q' key to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object
    video.release()

    # Destroy all windows
    cv2.destroyAllWindows()

    # Close the audio stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Close the output file
    output_file.close()
