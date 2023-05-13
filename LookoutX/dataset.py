import cv2  # OpenCV library
import time
import wave
import requests  # python http library
import pyaudio  # python audio library
import pyttsx3  # python text to speech library
# import speech_recognition as sr  # python speech recognition library
import numpy as np  # python numerical library
# import matplotlib.pyplot as plt
from pynput import keyboard
import wave
import whisper
import numpy as np
import sched
import sys

wifi_ip = '10.203.190.99'
port = '4747'

livestream_url = f'http://{wifi_ip}:{port}/video'

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()
model = whisper.load_model("base")

frames = []
video = cv2.VideoCapture(livestream_url)


def callback(in_data, frame_count, time_info, status):
    frames.append(in_data)
    return (in_data, pyaudio.paContinue)


class MyListener(keyboard.Listener):
    def __init__(self):
        super(MyListener, self).__init__(self.on_press, self.on_release)
        self.key_pressed = None
        self.wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        self.wf.setnchannels(CHANNELS)
        self.wf.setsampwidth(p.get_sample_size(FORMAT))
        self.wf.setframerate(RATE)

    def on_press(self, key):
        if key.char == 'r':
            self.key_pressed = True
        return True

    def on_release(self, key):
        if key.char == 'r':
            self.key_pressed = False
        return


listener = MyListener()
listener.start()
started = False
stream = None


def recorder():
    global started, p, stream, frames

    if (not listener.key_pressed) and (not started):
        # Read a frame from the video stream
        check, frame = video.read()

        # Waiting for 1ms
        key = cv2.waitKey(1)
        cv2.imshow('Frame', frame)

    if listener.key_pressed and not started:
        # Start the recording
        try:
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK,
                            stream_callback=callback)
            print("Stream active:", stream.is_active())
            started = True
            print("start Stream")
            check, frame = video.read()
            cv2.imwrite('frame.jpg', frame)
        except:
            raise

    elif not listener.key_pressed and started:
        print("Stop recording")
        stream.stop_stream()
        stream.close()
        p.terminate()
        listener.wf.writeframes(b''.join(frames))
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'rb')
        # I'd like to drop the middleman of writing to a file and just read from listener.wf, but that's tricky(?)
        audio_bytes = wf.readframes(wf.getnframes())
        listener.wf.close()
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
        result = model.transcribe(WAVE_OUTPUT_FILENAME)  # send the file to transcibe here
        print(result["text"])

        video.release()

        # Destroy all windows
        cv2.destroyAllWindows()

        # Close the audio stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        sys.exit()  # TODO: Don't exit, just restart the whole process after one query!
    # Reschedule the recorder function in 100 ms.
    task.enter(0.1, 1, recorder, ())


print("Press and hold the 'r' key to begin recording")
print("Release the 'r' key to end recording")
task = sched.scheduler(time.time, time.sleep)
task.enter(0.1, 1, recorder, ())
task.run()

# def stream(wifi_ip, port):

#     #output_audio_file = 'audio_output.wav'

#     # Initializing video


#     # Make a GET request to the DroidCam app to start the audio stream
#     #r = requests.get(audio_url, stream=True)
#     print("Press and hold the 'r' key to begin recording")
#     print("Release the 'r' key to end recording")
#     task = sched.scheduler(time.time, time.sleep)
#     task.enter(0.1, 1, recorder, ())
#     task.run()

#     # Initialize the flag variable


#     # Release the VideoCapture object
