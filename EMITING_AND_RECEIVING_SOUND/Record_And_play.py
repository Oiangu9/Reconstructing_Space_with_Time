import simpleaudio as sa
import numpy as np

'''
filename = 'myfile.wav'
wave_obj = sa.WaveObject.from_wave_file(filename)
play_obj = wave_obj.play()
play_obj.wait_done()  # Wait until sound has finished playing
'''

# REPRODUCIR NUMPY ARRAY
frequency = 440  # Our played note will be 440 Hz
fs = 44100  # 44100 samples per second
seconds = 1  # Note duration of 3 seconds

# Generate array with seconds*sample_rate steps, ranging between 0 and seconds
t = np.linspace(0, seconds, seconds * fs, False)

# Generate a 440 Hz sine wave
note = np.sin(frequency * t * 2 * np.pi)

# Ensure that highest value is in 16-bit range
audio = note * (2**15 - 1) / np.max(np.abs(note))
# Convert to 16-bit data
audio = audio.astype(np.int16)

# Start playback
play_obj = sa.play_buffer(audio, 1, 2, fs)

# Wait for playback to finish before exiting
play_obj.wait_done()
print("Done!")

# GRABAR A NUMPY ARRARY ######################################
import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 5  # Duration of recording
print("Recording...")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
print("Done")
write('output.wav', fs, myrecording)  # Save as WAV file



print("Now play!")
# Ensure that highest value is in 16-bit range
myrecording = myrecording * (2**15 - 1) / np.max(np.abs(myrecording))
# Convert to 16-bit data
myrecording = myrecording.astype(np.int16)

# Start playback
play_obj = sa.play_buffer(myrecording, 1, 2, fs)
play_obj.wait_done()
print("Done!")
