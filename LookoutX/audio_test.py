import os.path

import pyaudio
import wave
import whisper
import numpy as np

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

# stream = p.open(format=FORMAT,
#                 channels=CHANNELS,
#                 rate=RATE,
#                 input=True,
#                 frames_per_buffer=CHUNK)

# print("* recording")
#
# frames = []
#
# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#     data = stream.read(CHUNK)
#     frames.append(data)
#
# print("* done recording")
#
# stream.stop_stream()
# stream.close()
# p.terminate()
#
# wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# wf.setnchannels(CHANNELS)
# wf.setsampwidth(p.get_sample_size(FORMAT))
# wf.setframerate(RATE)
# wf.writeframes(b''.join(frames))
# wf.close()


# ----------------------------------------------------------------
# WHISPER TO RUN ON AUDIO FILE - https://github.com/openai/whisper

model = whisper.load_model("base")
# wf = wave.open(WAVE_OUTPUT_FILENAME, 'rb')
# # read the audio data as a byte string
# audio_bytes = wf.readframes(wf.getnframes())
# # close the WAV file
# wf.close()
# # convert the byte string to a numpy ndarray
# audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
# print(os.path.exists(WAVE_OUTPUT_FILENAME))
result = model.transcribe(WAVE_OUTPUT_FILENAME) # send the file to transcibe here
print(result["text"])
