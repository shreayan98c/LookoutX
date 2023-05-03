import os.path

import pyaudio
from pynput import keyboard
import wave
import whisper
import numpy as np
import sched
import sys
import time

wifi_ip = '172.20.20.20'
port = '4747'

livestream_url = f'http://{wifi_ip}:{port}/video'
audio_url = f'http://{wifi_ip}:{port}/audio.wav'

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()
model = whisper.load_model("base")

# stream = p.open(format=FORMAT,
#                 channels=CHANNELS,
#                 rate=RATE,
#                 input=True,
#                 frames_per_buffer=CHUNK)

# print("* recording")

frames = []

# Audio recording credit: https://stackoverflow.com/questions/44894796/pyaudio-and-pynput-recording-while-a-key-is-being-pressed-held-down
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

    if listener.key_pressed and not started:
        # Start the recording
        try:
            stream = p.open(format=FORMAT,
                             channels=CHANNELS,
                             rate=RATE,
                             input=True,
                             frames_per_buffer=CHUNK,
                             stream_callback = callback)
            print("Stream active:", stream.is_active())
            started = True
            print("start Stream")
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
        result = model.transcribe(WAVE_OUTPUT_FILENAME) # send the file to transcibe here
        print(result["text"])
        
        sys.exit() # TODO: Don't exit, just restart the whole process after one query!
    # Reschedule the recorder function in 100 ms.
    task.enter(0.1, 1, recorder, ())

print("Press and hold the 'r' key to begin recording")
print("Release the 'r' key to end recording")
task = sched.scheduler(time.time, time.sleep)
task.enter(0.1, 1, recorder, ())
task.run()

# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#     data = stream.read(CHUNK)
#     frames.append(data)

# print("* done recording")

# stream.stop_stream()
# stream.close()
# p.terminate()

# wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# wf.setnchannels(CHANNELS)
# wf.setsampwidth(p.get_sample_size(FORMAT))
# wf.setframerate(RATE)
# wf.writeframes(b''.join(frames))
# wf.close()


# ----------------------------------------------------------------
# WHISPER TO RUN ON AUDIO FILE - https://github.com/openai/whisper

# model = whisper.load_model("base")
# # wf = wave.open(WAVE_OUTPUT_FILENAME, 'rb')
# # # read the audio data as a byte string
# # audio_bytes = wf.readframes(wf.getnframes())
# # # close the WAV file
# # wf.close()
# # # convert the byte string to a numpy ndarray
# # audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
# # print(os.path.exists(WAVE_OUTPUT_FILENAME))
# result = model.transcribe(WAVE_OUTPUT_FILENAME) # send the file to transcibe here
# print(result["text"])
